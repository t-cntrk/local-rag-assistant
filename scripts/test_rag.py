"""Uçtan uca RAG için elle doğrulama scripti.

Embedding + chat modellerini BİR KEZ hazırlar, ardından hazır soruları sırayla
answer_query() ile geçirip cevabı ve kaynakları yazdırır. Son iki soru bilinçli
olarak belgelerde CEVABI OLMAYAN sorulardır; modelin "Bu bilgi elimdeki
belgelerde bulunmuyor." demesi beklenir.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe scripts/test_rag.py
"""

from __future__ import annotations

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

QUESTIONS = [
    ("Garanti süresi ne kadar?", "cevaplanabilir → 2 yıl demeli"),
    ("Kireç çözmeyi ne sıklıkla yapmalıyım?", "cevaplanabilir"),
    ("Wi-Fi'ye nasıl bağlanırım?", "cevaplanabilir"),
    ("Bu makine su geçirmez mi?", "cevaplanamaz → 'belgelerde bulunmuyor' demeli"),
    ("Fiyatı ne kadar?", "cevaplanamaz → 'belgelerde bulunmuyor' demeli"),
]


def main() -> None:
    print("=== UÇTAN UCA RAG TESTİ ===")
    print(f"Embed model: {config.EMBED_MODEL_ALIAS}")
    print(f"Chat model : {config.CHAT_MODEL_ALIAS}")
    print(f"top_k      : {config.TOP_K}\n")

    rag.prepare_models()

    try:
        for question, note in QUESTIONS:
            print("\n" + "=" * 78)
            print(f"SORU: {question}   [{note}]")
            print("=" * 78)

            start = time.time()
            result = rag.answer_query(question)
            elapsed = time.time() - start

            print(f"CEVAP:\n{result['answer']}")
            if result["answered"] and result["sources"]:
                print(f"\nKaynaklar: {', '.join(result['sources'])}")
            else:
                print("\n(Cevap bulunamadı — kaynak gösterilmiyor.)")
            print(
                "Getirilen parçalar: "
                + ", ".join(f"{src} ({score:.3f})" for score, src, _t, _c in result["results"])
            )
            print(f"Süre: {elapsed:.1f}s | finish_reason: {result['finish_reason']}")
    finally:
        foundry.shutdown()

    print("\nBitti.")


if __name__ == "__main__":
    main()
