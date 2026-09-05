"""Ölçüm: "bilmiyorum" güvencesi hedefli düzeltmelerden sonra ne kadar kararlı?

Önceki sürüm üç üretim ayarını (t=0.2 / t=0.0 / t=0.0+no_think) karşılaştırıyordu.
Ölçüm sonucu: t=0.0 uydurmayı kesiyor ama cevaplanabilir soruları yanlış reddediyor,
bu yüzden temperature=0.2 korundu. Bu sürüm yalnızca üretimdeki ayarı (t=0.2) ölçer
ve iki hedefli düzeltmenin etkisini gösterir:
  1. SYSTEM_PROMPT'a "özellik yokluğu" kuralı (6. kural)
  2. is_answered'ın anlamsal reddetme tanıma yeteneği

SKORLAMA: "doğru red" kararı rag.is_answered ile verilir (birebir cümle eşleşmesiyle
değil). Böylece modelin kendi kelimeleriyle kurduğu geçerli reddetmeler hata sayılmaz.

rag.py DEĞİŞTİRİLMEZ; answer_query() temperature'ı sabit kodladığı için chat çağrısı
burada kurulur, ancak SYSTEM_PROMPT / build_context / MAX_TOKENS / strip_thinking /
is_answered doğrudan rag'den alınır. Retrieval her soru için bir kez yapılır ve tüm
denemelerde aynı bağlam kullanılır.

Çalıştırma (proje kökünden, uzun sürer):
    .venv/Scripts/python.exe scripts/diag_hallucination.py
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

import config
import foundry
import rag
import retrieve

REPS = 5
TEMPERATURE = 0.2  # üretimdeki değer; ölçümle korunmasına karar verildi
THINK = True

QUESTIONS = [
    # --- Cevaplanamaz: doğru davranış = reddetmek ---
    {"q": "Bu makine su geçirmez mi?", "category": "cevaplanamaz"},
    {"q": "Fiyatı ne kadar?", "category": "cevaplanamaz"},
    {"q": "Kaç watt elektrik harcıyor?", "category": "cevaplanamaz"},
    # --- Cevaplanabilir: doğru davranış = beklenen bilgiyi vermek ---
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
        # Belgelerde karşılanıyor: "düşme sonucu oluşan hasarlar garanti kapsamı
        # dışındadır". Reddetmek burada HATADIR.
        "q": "Garanti düşme sonucu hasarı kapsıyor mu?",
        "category": "cevaplanabilir",
        "expect": re.compile(r"kapsam\w*\s+dış|kapsamınd[ae]\s+değil"),
    },
]


def generate(client, model_id: str, context: str, query: str) -> str:
    """rag.answer_query ile aynı prompt/bağlam."""
    user_message = f"BAĞLAM:\n{context}\n\nSoru: {query}"
    if not THINK:
        user_message += " /no_think"

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": rag.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=rag.MAX_TOKENS,
    )
    answer = rag.strip_thinking(response.choices[0].message.content or "")
    return answer or rag.FAILED_ANSWER_TEXT


def judge(item: dict, answer: str) -> bool:
    """Doğru davranış mı? Reddetme kararı rag.is_answered'a bırakılır."""
    answered = rag.is_answered(answer)
    if item["category"] == "cevaplanamaz":
        return not answered
    return answered and bool(item["expect"].search(rag._normalize(answer)))


def main() -> None:
    started = time.time()
    total_calls = len(QUESTIONS) * REPS

    print("=== ÖLÇÜM: hedefli düzeltmelerden sonra ===")
    print(f"Chat model : {config.CHAT_MODEL_ALIAS}")
    print(f"Ayar       : temperature={TEMPERATURE}, think={THINK}")
    print(f"top_k      : {config.TOP_K} | max_tokens: {rag.MAX_TOKENS}")
    print(f"Soru sayısı: {len(QUESTIONS)} | tekrar: {REPS} | toplam çağrı: {total_calls}\n")

    rag.prepare_models()
    client, model_id = foundry.get_model_client(config.CHAT_MODEL_ALIAS, verbose=False)

    results: dict[str, list[tuple[str, bool]]] = {}
    done = 0

    try:
        for item in QUESTIONS:
            query = item["q"]
            retrieved = retrieve.search(query, top_k=config.TOP_K)
            context = rag.build_context(retrieved)

            print("\n" + "=" * 78)
            print(f"SORU: {query}   [{item['category']}]")
            print("=" * 78)
            print("Bağlam: " + ", ".join(f"{src} ({s:.3f})" for s, src, _t, _c in retrieved))

            runs: list[tuple[str, bool]] = []
            for _rep in range(REPS):
                try:
                    answer = generate(client, model_id, context, query)
                except Exception as exc:
                    answer = f"HATA: {exc}"
                runs.append((answer, judge(item, answer)))
                done += 1

            results[query] = runs
            ok = sum(1 for _a, v in runs if v)
            print(f"  -> doğru {ok}/{REPS}   [{done}/{total_calls} çağrı]")
    finally:
        foundry.shutdown()

    # ------------------------------------------------------------------ tablo
    print("\n\n" + "=" * 78)
    print("ÖZET TABLO  (temperature=0.2, think=True)")
    print("=" * 78)
    qwidth = 48
    print("Soru".ljust(qwidth) + "kategori".ljust(16) + "doğru/5")
    print("-" * (qwidth + 24))
    for item in QUESTIONS:
        runs = results[item["q"]]
        ok = sum(1 for _a, v in runs if v)
        name = item["q"] if len(item["q"]) <= qwidth - 2 else item["q"][: qwidth - 5] + "..."
        print(name.ljust(qwidth) + item["category"].ljust(16) + f"{ok}/{REPS}")
    print("-" * (qwidth + 24))

    total_ok = sum(sum(1 for _a, v in results[i["q"]] if v) for i in QUESTIONS)
    print(f"TOPLAM: {total_ok}/{total_calls}")
    print("cevaplanamaz  : doğru = reddetti (is_answered False)")
    print("cevaplanabilir: doğru = cevap verdi VE beklenen bilgiyi içeriyor")

    # -------------------------------------------------------- hatalı cevaplar
    print("\n\n" + "=" * 78)
    print("HATALI DENEMELERİN TAM METNİ")
    print("=" * 78)
    any_bad = False
    for item in QUESTIONS:
        for i, (answer, verdict) in enumerate(results[item["q"]], start=1):
            if not verdict:
                any_bad = True
                kind = (
                    "UYDURMA / reddetmedi"
                    if item["category"] == "cevaplanamaz"
                    else "beklenen bilgiyi vermedi"
                )
                print(f"\n--- [{kind}] {item['q']} | deneme {i}")
                print(answer)
    if not any_bad:
        print("\n(Hatalı deneme yok.)")

    elapsed = time.time() - started
    print(f"\n\nToplam süre: {elapsed / 60:.1f} dakika ({elapsed:.0f} saniye), "
          f"{total_calls} çağrı, ortalama {elapsed / total_calls:.1f} s/çağrı")


if __name__ == "__main__":
    main()
