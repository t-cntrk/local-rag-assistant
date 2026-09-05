"""KahveUsta 300 yerel RAG destek asistanı — CLI.

Tamamen yerelde çalışır: belgeler SQLite'ta embedding'leriyle saklıdır, sorgu
Foundry Local üzerindeki embedding modeliyle vektörleştirilir, en alakalı
parçalar chat modeline bağlam olarak verilir.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe main.py

Çıkmak için: 'çıkış', 'exit' veya boş satır.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/ klasörünü içe aktarma yoluna ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
import foundry
import rag

EXIT_WORDS = {"çıkış", "cikis", "exit", "quit", "q"}


def main() -> None:
    print("=" * 70)
    print("  KahveUsta 300 — Yerel Destek Asistanı")
    print("=" * 70)
    print("Modeller hazırlanıyor, bu ilk açılışta biraz sürebilir...")
    print(f"  embedding: {config.EMBED_MODEL_ALIAS}")
    print(f"  chat     : {config.CHAT_MODEL_ALIAS}")

    try:
        rag.prepare_models()
    except Exception as exc:  # model indirme/yükleme hataları
        print(f"\nModeller hazırlanamadı: {exc}")
        raise SystemExit(1)

    print("\nHazır. Sorunuzu yazın. (Çıkmak için 'çıkış' veya boş satır)\n")

    try:
        while True:
            try:
                query = input("Soru> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query or query.lower() in EXIT_WORDS:
                break

            print("\nDüşünüyorum...", end="\r", flush=True)
            try:
                result = rag.answer_query(query)
            except Exception as exc:
                print(f"Hata: {exc}          \n")
                continue

            print(" " * 20, end="\r")  # "Düşünüyorum..." satırını temizle
            print(f"\n{result['answer']}")
            # Reddedilen sorularda kaynak gösterilmez (rag.answer_query kararı).
            if result["answered"] and result["sources"]:
                print(f"\nKaynaklar: {', '.join(result['sources'])}")
            print()
    finally:
        foundry.shutdown()

    print("Görüşmek üzere.")


if __name__ == "__main__":
    main()
