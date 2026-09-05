"""Ölçüm: sorgu genişletme uydurma güvencesini bozuyor mu? (uçtan uca)

Bağlam: scripts/diag_query_expansion.py, retrieval'a ürün bağlamı öneki eklemenin
bozuk renk sorgularını düzelttiğini gösterdi. Ama aynı genişletme "Bu makine su
geçirmez mi?" sorusunun getirdiği chunk kümesini de değiştirdi. Prompt'a dayalı
reddetme güvencesi (A modunda cevaplanamaz sorularda 15/15) o yeni bağlamda hâlâ
tutuyor mu — bu script onu ölçer.

İKİ MOD
    (A) mevcut       : sorgu olduğu gibi embed edilir (bugünkü üretim davranışı)
    (B) genişletilmiş: retrieval için sorgunun başına ürün bağlamı öneki eklenir

ÖNEMLİ: Genişletme YALNIZCA retrieval'ı etkiler. Modele giden asıl soru her iki
modda da kullanıcının yazdığı sorudur; yalnızca hangi chunk'ların bağlama gireceği
değişir. Genişletme yöntemi diag_query_expansion.py'deki B ile birebir aynıdır.

DEĞİŞTİRİLMEYENLER: SYSTEM_PROMPT, temperature=0.2, MAX_TOKENS, top_k, model.
Hepsi rag/config'ten alınır. Bu script salt okur; kalıcı dosya değiştirmez.

Doğru/yanlış kararı rag.is_refusal (anlamsal) ile verilir, birebir cümleyle değil.

Çalıştırma (proje kökünden, uzun sürer — 70 model çağrısı):
    .venv/Scripts/python.exe scripts/diag_expansion_hallucination.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

import config
import foundry
import rag
import retrieve

REPS = 5
TEMPERATURE = 0.2  # üretimdeki değer — değiştirilmez
PRODUCT_CONTEXT = "KahveUsta 300 akıllı kahve makinesi hakkında soru: "

QUESTIONS = [
    # --- Cevaplanamaz: doğru davranış = reddetmek (asıl sınav) ---
    {"q": "Bu makine su geçirmez mi?", "category": "cevaplanamaz"},
    {"q": "Fiyatı ne kadar?", "category": "cevaplanamaz"},
    {"q": "Kaç watt elektrik harcıyor?", "category": "cevaplanamaz"},
    # --- Cevaplanabilir: bozulmamalı ---
    {
        "q": "Garanti süresi ne kadar?",
        "category": "cevaplanabilir",
        "expect": re.compile(r"2 yıl|24 ay"),
    },
    {
        "q": "Kireç çözmeyi ne sıklıkla yapmalıyım?",
        "category": "cevaplanabilir",
        "expect": re.compile(r"3 ayda|300 fincan"),
    },
    {
        "q": "Garanti düşme sonucu hasarı kapsıyor mu?",
        "category": "cevaplanabilir",
        "expect": re.compile(r"kapsam\w*\s+dış|kapsamınd[ae]\s+değil"),
    },
    {
        # B'nin asıl faydası: A'da retrieval renk chunk'ını getiremediği için
        # modelin "bulunmuyor" demesi BEKLENİR (doğru davranış, ama eksik cevap).
        "q": "renkler neler",
        "category": "cevaplanabilir",
        "expect": re.compile(r"paslanmaz çelik|mat siyah"),
    },
]


def retrieve_plain(query: str) -> list[tuple[float, str, str, str]]:
    """(A) Bugünkü üretim davranışı."""
    return retrieve.search(query, top_k=config.TOP_K)


def retrieve_expanded(query: str, client, embed_model_id: str
                      ) -> list[tuple[float, str, str, str]]:
    """(B) Ürün bağlamı öneki YALNIZCA retrieval sorgusuna eklenir."""
    chunks, normalized = retrieve.load_index()
    formatted = retrieve.format_query(f"{PRODUCT_CONTEXT}{query}")
    vector = np.array(foundry.embed_text(client, embed_model_id, formatted), dtype=np.float32)
    vector /= np.linalg.norm(vector)
    scores = normalized @ vector
    order = np.argsort(-scores)[: config.TOP_K]
    return [
        (float(scores[i]), chunks[i]["source"], chunks[i]["title"], chunks[i]["content"])
        for i in order
    ]


def generate(client, chat_model_id: str, context: str, query: str) -> str:
    """rag.answer_query ile aynı prompt, aynı temperature. Soru ORİJİNAL hâliyle."""
    response = client.chat.completions.create(
        model=chat_model_id,
        messages=[
            {"role": "system", "content": rag.SYSTEM_PROMPT},
            {"role": "user", "content": f"BAĞLAM:\n{context}\n\nSoru: {query}"},
        ],
        temperature=TEMPERATURE,
        max_tokens=rag.MAX_TOKENS,
    )
    answer = rag.strip_thinking(response.choices[0].message.content or "")
    return answer or rag.FAILED_ANSWER_TEXT


def judge(item: dict, answer: str) -> bool:
    """Reddetme kararı rag.is_refusal (anlamsal) ile verilir."""
    refused = rag.is_refusal(answer)
    if item["category"] == "cevaplanamaz":
        return refused
    return (not refused) and bool(item["expect"].search(rag._normalize(answer)))


def main() -> None:
    started = time.time()
    modes = ["A: mevcut", "B: genişletilmiş"]
    total_calls = len(QUESTIONS) * len(modes) * REPS

    print("=== ÖLÇÜM: sorgu genişletme + uydurma güvencesi (uçtan uca) ===")
    print(f"Chat model : {config.CHAT_MODEL_ALIAS}")
    print(f"Embed model: {config.EMBED_MODEL_ALIAS}")
    print(f"Ayar       : temperature={TEMPERATURE} | top_k={config.TOP_K} "
          f"| max_tokens={rag.MAX_TOKENS}")
    print(f"Genişletme : {PRODUCT_CONTEXT!r} (yalnızca retrieval)")
    print(f"Toplam çağrı: {total_calls}\n")

    rag.prepare_models()
    embed_client, embed_model_id = foundry.get_model_client(
        config.EMBED_MODEL_ALIAS, verbose=False
    )
    chat_client, chat_model_id = foundry.get_model_client(
        config.CHAT_MODEL_ALIAS, verbose=False
    )

    # results[soru][mod] = list[(answer, verdict)]
    results: dict[str, dict[str, list[tuple[str, bool]]]] = {}
    done = 0

    try:
        for item in QUESTIONS:
            query = item["q"]
            results[query] = {}

            print("\n" + "=" * 78)
            print(f"SORU: {query}   [{item['category']}]")
            print("=" * 78)

            for mode in modes:
                if mode.startswith("A"):
                    retrieved = retrieve_plain(query)
                else:
                    retrieved = retrieve_expanded(query, embed_client, embed_model_id)
                context = rag.build_context(retrieved)

                print(f"\n  [{mode}] getirilen bağlam:")
                for rank, (score, source, title, _c) in enumerate(retrieved, start=1):
                    print(f"    {rank}. {score:.4f} | {source} | {title}")

                runs: list[tuple[str, bool]] = []
                for _rep in range(REPS):
                    try:
                        answer = generate(chat_client, chat_model_id, context, query)
                    except Exception as exc:
                        answer = f"HATA: {exc}"
                    runs.append((answer, judge(item, answer)))
                    done += 1

                results[query][mode] = runs
                ok = sum(1 for _a, v in runs if v)
                print(f"    -> doğru {ok}/{REPS}   [{done}/{total_calls} çağrı]")
    finally:
        foundry.shutdown()

    # ------------------------------------------------------------------ tablo
    print("\n\n" + "=" * 78)
    print("KARŞILAŞTIRMA TABLOSU  (temperature=0.2)")
    print("=" * 78)
    qwidth = 42
    print("Soru".ljust(qwidth) + "kategori".ljust(16) + "A".rjust(6) + "B".rjust(8))
    print("-" * (qwidth + 30))
    for item in QUESTIONS:
        cells = []
        for mode in modes:
            ok = sum(1 for _a, v in results[item["q"]][mode] if v)
            cells.append(f"{ok}/{REPS}")
        name = item["q"] if len(item["q"]) <= qwidth - 2 else item["q"][: qwidth - 5] + "..."
        print(name.ljust(qwidth) + item["category"].ljust(16)
              + cells[0].rjust(6) + cells[1].rjust(8))
    print("-" * (qwidth + 30))

    # --------------------------------------------------- asıl sorunun cevabı
    print("\nCEVAPLANAMAZ SORULARIN TOPLAMI (asıl sınav):")
    for mode in modes:
        total = 0
        for item in QUESTIONS:
            if item["category"] != "cevaplanamaz":
                continue
            total += sum(1 for _a, v in results[item["q"]][mode] if v)
        cap = 3 * REPS
        print(f"  {mode:18s} -> {total}/{cap}"
              + ("   (güvence korundu)" if total == cap else "   (GERİLEME)"))

    print("\nCEVAPLANABİLİR SORULARIN TOPLAMI:")
    for mode in modes:
        total = 0
        for item in QUESTIONS:
            if item["category"] != "cevaplanabilir":
                continue
            total += sum(1 for _a, v in results[item["q"]][mode] if v)
        print(f"  {mode:18s} -> {total}/{4 * REPS}")

    # -------------------------------------------------------- hatalı cevaplar
    print("\n\n" + "=" * 78)
    print("HATALI DENEMELERİN TAM METNİ")
    print("=" * 78)
    any_bad = False
    for item in QUESTIONS:
        for mode in modes:
            for i, (answer, verdict) in enumerate(results[item["q"]][mode], start=1):
                if verdict:
                    continue
                any_bad = True
                kind = (
                    "UYDURMA / reddetmedi"
                    if item["category"] == "cevaplanamaz"
                    else "beklenen bilgiyi vermedi"
                )
                print(f"\n--- [{kind}] {item['q']} | {mode} | deneme {i}")
                print(answer)
    if not any_bad:
        print("\n(Hatalı deneme yok.)")

    elapsed = time.time() - started
    print(f"\n\nToplam süre: {elapsed / 60:.1f} dakika ({elapsed:.0f} saniye), "
          f"{total_calls} çağrı, ortalama {elapsed / total_calls:.1f} s/çağrı")


if __name__ == "__main__":
    main()
