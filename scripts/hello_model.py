"""Minimal 'hello model' testi — Foundry Local Python SDK (Windows/WinML).

Bir sohbet modelini indirir, yükler, tek bir kısa tamamlama üretir ve yazdırır.
RAG mantığı burada YOK; bu yalnızca ortam doğrulama scriptidir.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe scripts/hello_model.py
"""

import sys

# Türkçe karakterlerin terminalde düzgün görünmesi için stdout'u UTF-8'e sabitle.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Model sabit kodlanmadı; alias burada, tek bir değişkende tutuluyor.
# 'foundry model list' çıktısındaki herhangi bir Chat modeliyle değiştirilebilir
# (ör. "phi-3.5-mini", "qwen3-0.6b", "qwen2.5-1.5b").
MODEL_ALIAS = "qwen2.5-0.5b"

PROMPT = "Tek cümlede kendini tanıt ve hangi model olduğunu söyle."

from foundry_local_sdk import Configuration, FoundryLocalManager


def main() -> None:
    # 1) SDK'yı başlat (yerel Foundry servisi ayağa kalkar).
    config = Configuration(app_name="local_rag_hello")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2) Modeli alias ile katalogdan seç.
    model = manager.catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise SystemExit(
            f"Model bulunamadı: {MODEL_ALIAS!r}. "
            "'foundry model list' ile geçerli bir alias seçin."
        )

    # 3) Gerekiyorsa indir (zaten cache'liyse atlanır).
    print(f"Model hazırlanıyor: {MODEL_ALIAS}")

    def on_progress(percent: float) -> None:
        print(f"\r  indiriliyor: {percent:6.2f}%", end="", flush=True)

    model.download(progress_callback=on_progress)
    print()

    # 4) Belleğe yükle.
    print("Model yükleniyor...")
    model.load()

    # 5) Tek bir kısa tamamlama üret.
    client = model.get_chat_client()
    response = client.complete_chat([{"role": "user", "content": PROMPT}])
    answer = response.choices[0].message.content

    print("\n=== Model cevabı ===")
    print(answer)
    print("====================")

    # 6) Temizlik.
    model.unload()


if __name__ == "__main__":
    main()
