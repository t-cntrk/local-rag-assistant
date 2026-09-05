"""Teşhis: "renkler neler" sorusu neden cevaplanamıyor?

Renk bilgisi (paslanmaz çelik / mat siyah) belgelerde VAR ama asistan soruyu
reddediyor. Bu script, sorunun retrieval katmanında mı yoksa cevap üretiminde mi
olduğunu gösterir: üç farklı sorgu varyantı için getirilen top-3 chunk'ı ve renk
chunk'ının tüm sıralamadaki yerini yazdırır.

SADECE OKUR — veritabanını, prompt'u veya ayarları değiştirmez.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe scripts/diag_colors.py
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

import numpy as np

import config
import foundry
import retrieve

PREVIEW_CHARS = 150

QUERIES = [
    "renkler neler",
    "Hangi renkler var",
    "renk seçenekleri neler",
]

# Renk bilgisinin geçtiği chunk'ı bulmak için aranan ifadeler.
COLOR_MARKERS = ("mat siyah", "paslanmaz çelik")


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def find_color_chunk(chunks: list[dict]) -> int | None:
    """Renk bilgisini içeren chunk'ın indeksini döndürür."""
    for i, ch in enumerate(chunks):
        lowered = ch["content"].casefold()
        if all(marker.casefold() in lowered for marker in COLOR_MARKERS):
            return i
    return None


def main() -> None:
    print("=== TEŞHİS: renk sorusu ===")
    print(f"Veritabanı : {config.DB_PATH}")
    print(f"Embed model: {config.EMBED_MODEL_ALIAS}")
    print(f"top_k      : {config.TOP_K}\n")

    # Embedding modelini bir kez hazırla.
    foundry.get_model_client(config.EMBED_MODEL_ALIAS)

    chunks, normalized = retrieve.load_index()
    print(f"Toplam chunk: {len(chunks)}")

    color_idx = find_color_chunk(chunks)
    if color_idx is None:
        print("\nUYARI: Renk bilgisi içeren chunk BULUNAMADI.")
        target = None
    else:
        target = chunks[color_idx]
        print("\n--- Hedef (renk bilgisi içeren) chunk ---")
        print(f"source : {target['source']}")
        print(f"title  : {target['title']}")
        print(f"content: {preview(target['content'], 300)}")

    try:
        for query in QUERIES:
            print("\n" + "=" * 78)
            print(f"SORGU: {query!r}")
            print("=" * 78)

            results = retrieve.search(query, top_k=config.TOP_K)

            print(f"Getirilen top-{config.TOP_K}:")
            for rank, (score, source, title, content) in enumerate(results, start=1):
                print(f"  {rank}. skor={score:.4f} | {source} | {title}")
                print(f"     {preview(content)}")

            if target is None:
                continue

            # Hedef chunk getirilenler arasında mı?
            hit = any(
                source == target["source"] and title == target["title"]
                for _s, source, title, _c in results
            )
            print(
                f"\n  >> Renk chunk'ı ('{target['source']} — {target['title']}') "
                f"top-{config.TOP_K} içinde: {'EVET' if hit else 'HAYIR'}"
            )

            # Getirilmediyse bile: tüm sıralamada kaçıncı ve skoru ne?
            query_vector = np.array(retrieve.embed_query(query), dtype=np.float32)
            query_vector /= np.linalg.norm(query_vector)
            scores = normalized @ query_vector
            order = np.argsort(-scores)
            position = int(np.where(order == color_idx)[0][0]) + 1
            print(
                f"  >> Renk chunk'ının tüm sıralamadaki yeri: {position}/{len(chunks)} "
                f"(skor {float(scores[color_idx]):.4f}); "
                f"top-{config.TOP_K} eşiği {float(scores[order[config.TOP_K - 1]]):.4f}"
            )
    finally:
        foundry.shutdown()

    print("\nBitti.")


if __name__ == "__main__":
    main()
