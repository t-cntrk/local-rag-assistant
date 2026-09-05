"""Retrieval katmanı için elle doğrulama scripti.

Embedding modelini BİR KEZ yükler, ardından hazır soruları sırayla çalıştırıp
her biri için en iyi 3 chunk'ı skor + source + title + içerik önizlemesi ile
yazdırır. Son üç soru bilinçli olarak belgelerde CEVABI OLMAYAN sorular; skor
farkının (yüksek vs. düşük benzerlik) gözlemlenmesi amaçlanır.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe scripts/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
import foundry
import retrieve

PREVIEW_CHARS = 100

QUESTIONS = [
    ("Garanti süresi ne kadar?", "cevaplanabilir"),
    ("Kireç çözmeyi ne sıklıkla yapmalıyım?", "cevaplanabilir"),
    ("Wi-Fi'ye nasıl bağlanırım?", "cevaplanabilir"),
    ("Bu makine su geçirmez mi?", "cevaplanamaz — belgelerde yok"),
    ("Fiyatı ne kadar?", "cevaplanamaz — belgelerde yok"),
]


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """İçeriği tek satıra indirger ve kısaltır."""
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def main() -> None:
    print("=== RETRIEVAL TESTİ ===")
    print(f"Veritabanı : {config.DB_PATH}")
    print(f"Embed model: {config.EMBED_MODEL_ALIAS}")
    print(f"top_k      : 3\n")

    # Embedding modelini bir kez yükle (sonraki sorgular aynı servisi kullanır).
    foundry.get_model_client(config.EMBED_MODEL_ALIAS)

    chunks, _ = retrieve.load_index()
    print(f"Yüklenen chunk sayısı: {len(chunks)}")
    print(f"\nSorgu instruction biçimi:\n  {retrieve.format_query('<soru>')!r}")

    try:
        for question, note in QUESTIONS:
            print("\n" + "=" * 78)
            print(f"SORU: {question}   [{note}]")
            print("=" * 78)
            for rank, (score, source, title, content) in enumerate(
                retrieve.search(question, top_k=3), start=1
            ):
                print(f"  {rank}. skor={score:.4f} | {source} | {title}")
                print(f"     {preview(content)}")
    finally:
        foundry.shutdown()

    print("\nBitti.")


if __name__ == "__main__":
    main()
