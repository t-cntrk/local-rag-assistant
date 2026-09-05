"""RAG orkestrasyonu: retrieval sonuçlarını bağlama çevir -> chat modeliyle cevapla.

Akış:
    soru -> retrieve.search() -> build_context() -> system + user mesajı
         -> chat modeli (qwen3-1.7b) -> cevap + kaynaklar

Tasarım notları:
- **Skor eşiği kullanılmaz.** "Bilmiyorum" güvencesi tamamen SYSTEM_PROMPT ile
  sağlanır: bağlamda olmayan bir şey sorulduğunda model sabit bir cümle döndürür.
- qwen3-1.7b hibrit bir "thinking" modelidir; düşünme çıktısı ayrı bir alanda
  değil, doğrudan ``message.content`` içinde ``<think>...</think>`` bloğu olarak
  gelir (empirik doğrulandı). Bu blok kullanıcıya gösterilmeden ayıklanır.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# src/ klasörünü içe aktarma yoluna ekle (config.py, retrieve.py, foundry.py komşular).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import foundry
import retrieve

# İki farklı reddetme durumu, iki farklı sabit cümle:
#   - Ürünle ilgili ama belgelerde olmayan bilgi  -> NO_ANSWER_TEXT
#   - Ürünle tamamen alakasız istek (kapsam dışı) -> OUT_OF_SCOPE_TEXT
NO_ANSWER_TEXT = "Bu bilgi elimdeki belgelerde bulunmuyor."
OUT_OF_SCOPE_TEXT = "Ben yalnızca KahveUsta 300 hakkında yardımcı olabilirim."

SYSTEM_PROMPT = (
    "Sen KahveUsta 300 akıllı kahve makinesi destek asistanısın. "
    "SADECE sana verilen bağlam parçalarındaki bilgiyi kullanarak Türkçe cevap ver. "
    "YALNIZCA KahveUsta 300 ürünü hakkında bilgi verirsin.\n"
    "\n"
    "Kurallar:\n"
    "1. Önce isteğin KAPSAMDA olup olmadığına bak. KahveUsta 300 ürünüyle ilgisi "
    "olmayan istekler kapsam DIŞIDIR: masal/hikâye/şiir anlatma veya yazma, şaka, "
    "genel kültür ve haber soruları, kod yazma, matematik, sohbet, kişisel tavsiye "
    "gibi. Böyle bir istekte kibarca reddet ve tam olarak şunu yaz, başka hiçbir "
    f"şey yazma: {OUT_OF_SCOPE_TEXT}\n"
    "2. İstek ürünle ilgiliyse cevabı doğrudan ver. Soruyu tekrar etme, bağlamı "
    "olduğu gibi kopyalama, başlık veya 'Sonuç' bölümü ekleme.\n"
    "3. Cevap en fazla 3 cümle olsun; adım listesi gerekiyorsa en fazla 5 kısa madde.\n"
    "4. Bağlamda soruyu karşılayan bir bilgi VARSA onu net biçimde söyle "
    "(sayı, süre ve koşulları bağlamdaki gibi aktar).\n"
    "5. İstek ürünle ilgili ama bağlamda soruyu karşılayan bilgi YOKSA tam olarak "
    f"şunu yaz ve başka hiçbir şey yazma: {NO_ANSWER_TEXT}\n"
    "6. Bir özelliğin veya niteliğin var olup olmadığı sorulduğunda (örn. 'su "
    "geçirmez mi?', 'X özelliği var mı?') ve verilen bağlam parçalarında o "
    "özellikten hiç söz edilmiyorsa, özelliğin olup olmadığına dair YORUM YAPMA "
    "veya TAHMİN YÜRÜTME. Bir özelliğin bağlamda geçmemesi, o özelliğin olmadığı "
    f"anlamına gelmez. Böyle durumlarda doğrudan şunu yaz: {NO_ANSWER_TEXT}\n"
    "7. Bilgi UYDURMA, tahmin yürütme, bağlam dışına çıkma. Bağlamda geçmeyen "
    "bir özelliği ne doğrula ne de reddet. Ürün bağlamını konu dışı bir isteğe "
    "uydurmaya ÇALIŞMA (ör. kahve makinesi belgelerinden masal veya şiir türetme).\n"
    "\n"
    "İki reddetme cümlesini birbirine karıştırma: kapsam dışı istek → 1. kuraldaki "
    "cümle; ürünle ilgili ama belgelerde bulunmayan bilgi → 5. ve 6. kuraldaki cümle."
)

# Düşünme bloğu: hem kapanmış hem (token limiti nedeniyle) kapanmamış hâli.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)

# Cevap için üst sınır. Düşünme bloğu da bu bütçeden harcandığı için geniş
# tutulur: 1024'te düşünme adımı bütçeyi tüketip cevabı tamamen boş bırakıyordu,
# 3072'de ise uzun düşünen sorularda nihai cevap kelime ortasında kesilebiliyordu.
MAX_TOKENS = 4096

# Model cevap üretemediğinde (ör. düşünme adımı token bütçesini tüketti).
FAILED_ANSWER_TEXT = (
    "(Model bu soru için cevap üretemedi — düşünme adımı token sınırına takıldı.)"
)


def strip_thinking(text: str) -> str:
    """Model çıktısındaki <think>...</think> bloğunu ayıklar."""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)  # kapanmamış blok kalıntısı
    return cleaned.strip()


def _normalize(text: str) -> str:
    """Karşılaştırma için metni sadeleştirir (markdown kalın işaretleri, boşluk).

    Not: Türkçe "İ" harfi casefold edildiğinde "i" + U+0307 (birleşik nokta)
    üretir; bu görünmez işaret desen eşleşmesini bozduğu için temizlenir.
    """
    folded = text.replace("*", "").replace("_", "").casefold().replace("̇", "")
    return " ".join(folded.split())


# Modelin sabit cümle yerine kendi kelimeleriyle kurduğu reddetme kalıpları.
# Kasıtlı olarak DAR tutulur: her kalıp bir "bilgi/belge" ismini bir olumsuzlama
# ile aynı cümlecikte arar. Böylece "düşme sonucu oluşan hasarlar garanti
# kapsamı dışındadır" gibi GERÇEK cevaplar yanlışlıkla reddetme sayılmaz.
_REFUSAL_PATTERNS = (
    # "... belgelerde bulunmuyor / yer almıyor / geçmiyor / yok"
    re.compile(
        r"belge\w*\s+(?:\w+\s+){0,4}?"
        r"(?:bulunmuyor|bulunmamaktadır|yer almıyor|yer almamaktadır|geçmiyor|"
        r"mevcut değil|yok\b)"
    ),
    # "bu bilgi ... yok / bulunmuyor / belirtilmemiştir"
    re.compile(
        r"bu bilgi\w*\s+(?:\w+\s+){0,4}?"
        r"(?:bulunmuyor|bulunmamaktadır|yer almıyor|mevcut değil|belirtilmemiş\w*|yok\b)"
    ),
    # "... bilgi(si) bulunmuyor / yer almıyor / içermez / mevcut değil"
    re.compile(
        r"bilgi\w*\s+(?:\w+\s+){0,3}?"
        r"(?:bulunmuyor|bulunmamaktadır|yer almıyor|yer almamaktadır|içermez|"
        r"mevcut değil)"
    ),
    # Kapsam dışı reddi, sabit cümleden sapsa bile.
    re.compile(r"yalnızca kahveusta 300 hakkında"),
)


def is_refusal(answer: str) -> bool:
    """Cevap, bir reddetme (bilgi yok / kapsam dışı / üretilemedi) mi?

    Önce sabit cümleler, sonra anlamsal kalıplar aranır. Model reddi kendi
    kelimeleriyle kurduğunda ("fiyat bilgisi belgelerde yer almıyor") birebir
    eşleşme kaçırıyordu; kalıplar bu boşluğu kapatır.
    """
    normalized = _normalize(answer)
    if not normalized:
        return True
    if any(
        _normalize(marker) in normalized
        for marker in (NO_ANSWER_TEXT, OUT_OF_SCOPE_TEXT, FAILED_ANSWER_TEXT)
    ):
        return True
    return any(pattern.search(normalized) for pattern in _REFUSAL_PATTERNS)


def is_answered(answer: str) -> bool:
    """Cevap gerçek bir bilgi mi, yoksa reddetme/başarısızlık cümlesi mi?

    Reddetmenin her biçimi (belgelerde yok / kapsam dışı / model üretemedi)
    answered=False sayılır — hiçbirinde kaynak gösterilmez.
    """
    return not is_refusal(answer)


def build_context(results: list[tuple[float, str, str, str]]) -> str:
    """Retrieval sonuçlarını numaralı, kaynaklı bir bağlam metnine çevirir.

    Biçim:
        [1] (Kaynak: 04_temizlik_bakim.txt — Kireç Çözme)
        <içerik>
    """
    blocks: list[str] = []
    for i, (_score, source, title, content) in enumerate(results, start=1):
        blocks.append(f"[{i}] (Kaynak: {source} — {title})\n{content}")
    return "\n\n".join(blocks)


def unique_sources(results: list[tuple[float, str, str, str]]) -> list[str]:
    """Getirilen parçaların benzersiz dosya adlarını (sırayı koruyarak) döndürür."""
    seen: list[str] = []
    for _score, source, _title, _content in results:
        if source not in seen:
            seen.append(source)
    return seen


def answer_query(
    query: str,
    top_k: int = config.TOP_K,
    think: bool = True,
    model_alias: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Bir soruyu uçtan uca cevaplar.

    Args:
        query: Kullanıcının sorusu.
        top_k: Bağlama alınacak chunk sayısı.
        think: False ise Qwen3'ün ``/no_think`` yumuşak anahtarı kullanılır
            (daha hızlı ama küçük modelde belirgin şekilde daha isabetsiz).
        model_alias: Chat modelini geçici olarak değiştirmek için (karşılaştırma
            testleri). None ise config.CHAT_MODEL_ALIAS kullanılır.
        verbose: Model hazırlama çıktısını göster.

    Returns:
        {
            "answer":        str,        # kullanıcıya gösterilecek cevap
            "answered":      bool,       # False ise cevap reddetme/başarısızlık cümlesi
            "sources":       list[str],  # answered False ise boş liste
            "results":       list[tuple],# (score, source, title, content) — hata ayıklama
            "finish_reason": str,        # "length" ise cevap token sınırında kesildi
        }
    """
    results = retrieve.search(query, top_k=top_k, verbose=verbose)
    context = build_context(results)

    user_message = f"BAĞLAM:\n{context}\n\nSoru: {query}"
    if not think:
        user_message += " /no_think"

    alias = model_alias or config.CHAT_MODEL_ALIAS
    client, model_id = foundry.get_model_client(alias, verbose=verbose)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )

    choice = response.choices[0]
    # "length" => üretim MAX_TOKENS'a takıldı, yani cevap kelime ortasında
    # kesilmiş olabilir. Teşhis için dönüş sözlüğünde taşınır.
    finish_reason = choice.finish_reason

    raw = choice.message.content or ""
    answer = strip_thinking(raw)
    if not answer:
        # Token bütçesi düşünme bloğunda tükenmiş olabilir.
        answer = FAILED_ANSWER_TEXT

    # Model "bilmiyorum" dediyse hiçbir kaynağı fiilen kullanmamıştır; kaynak
    # göstermek yanıltıcı olur. Bu karar tek yerde verilir, arayüzler sadece
    # "answered" alanına bakar. Ham skorlu sonuçlar hata ayıklama için kalır.
    answered = is_answered(answer)

    return {
        "answer": answer,
        "answered": answered,
        "sources": unique_sources(results) if answered else [],
        "results": results,
        "finish_reason": finish_reason,
    }


def prepare_models(verbose: bool = True) -> None:
    """Embedding ve chat modellerini önceden yükler (ilk sorunun beklemesini azaltır)."""
    foundry.get_model_client(config.EMBED_MODEL_ALIAS, verbose=verbose)
    foundry.get_model_client(config.CHAT_MODEL_ALIAS, verbose=verbose)
