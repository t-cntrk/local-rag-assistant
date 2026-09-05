"""Foundry Local için ortak yardımcılar.

Hem ingestion (belge embedding'i) hem de retrieval (sorgu embedding'i) hem de
ileride cevap üretimi aynı üç adıma ihtiyaç duyar:

    1. SDK'yı başlat (singleton FoundryLocalManager)
    2. Modeli indir + belleğe yükle
    3. OpenAI-uyumlu web servisine bağlan ve *çözümlenmiş* model id'sini kullan

Bu modül o adımları tek yerde toplar ve **idempotent** yapar: web servisi bir kez
başlatılır, aynı alias ikinci kez istendiğinde tekrar indirilip yüklenmez.

Tipik kullanım:

    import foundry, config

    client, model_id = foundry.get_model_client(config.EMBED_MODEL_ALIAS)
    vectors = foundry.embed_texts(client, model_id, ["merhaba", "dünya"])
    ...
    foundry.shutdown()

ya da context manager ile:

    with foundry.foundry_model(config.EMBED_MODEL_ALIAS) as (client, model_id):
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from foundry_local_sdk import Configuration, FoundryLocalManager
from openai import OpenAI

# Foundry Local'a kendimizi tanıtırken kullanılan ad.
APP_NAME = "local_rag"

# --- Modül içi önbellek (süreç ömrü boyunca) --------------------------------- #
_client: OpenAI | None = None
_loaded_models: dict[str, str] = {}  # alias -> çözümlenmiş model id
_batch_embeddings_supported: bool | None = None  # None = henüz denenmedi


def get_manager() -> FoundryLocalManager:
    """SDK singleton'ını döndürür; gerekiyorsa başlatır.

    ``FoundryLocalManager.initialize()`` ikinci kez çağrılırsa SDK hata fırlatır,
    bu yüzden önce mevcut instance kontrol edilir.
    """
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    return FoundryLocalManager.instance


def get_client(verbose: bool = True) -> OpenAI:
    """OpenAI-uyumlu istemciyi döndürür; web servisini gerekiyorsa başlatır."""
    global _client

    manager = get_manager()
    if manager.urls is None:
        manager.start_web_service()

    if _client is None:
        base_url = manager.urls[0].rstrip("/") + "/v1"
        if verbose:
            print(f"OpenAI-uyumlu uç nokta: {base_url}")
        _client = OpenAI(base_url=base_url, api_key="not-needed")

    return _client


def load_model(alias: str, verbose: bool = True) -> str:
    """Modeli indirir + yükler ve *çözümlenmiş* model id'sini döndürür.

    Çözümlenmiş id, varyant/EP bilgisini içerir (ör.
    ``qwen3-embedding-0.6b-generic-cpu:1``) ve API çağrılarında alias yerine bu
    kullanılmalıdır. Aynı alias için sonuç önbelleğe alınır.
    """
    if alias in _loaded_models:
        return _loaded_models[alias]

    manager = get_manager()
    model = manager.catalog.get_model(alias)
    if model is None:
        raise SystemExit(f"Model bulunamadı: {alias!r}")

    if verbose:
        print(f"Model hazırlanıyor: {alias}")
    model.download(
        progress_callback=(
            (lambda p: print(f"\r  indiriliyor: {p:6.2f}%", end="", flush=True))
            if verbose
            else None
        )
    )
    if verbose:
        print()
    model.load()

    _loaded_models[alias] = model.id
    if verbose:
        print(f"Çözümlenmiş model id: {model.id}")
    return model.id


def get_model_client(alias: str, verbose: bool = True) -> tuple[OpenAI, str]:
    """Bir alias için (OpenAI istemcisi, çözümlenmiş model id) çifti döndürür."""
    model_id = load_model(alias, verbose=verbose)
    client = get_client(verbose=verbose)
    return client, model_id


def embed_texts(client: OpenAI, model_id: str, texts: list[str]) -> list[list[float]]:
    """Metin listesi için embedding vektörlerini (girdi sırasıyla) döndürür.

    Önce tek istekte toplu (batch) gönderim denenir; uç nokta liste girdisini
    desteklemiyorsa metin başına tek tek istek atılır ve bu tercih hatırlanır.
    """
    global _batch_embeddings_supported

    if not texts:
        return []

    if _batch_embeddings_supported is not False:
        try:
            resp = client.embeddings.create(model=model_id, input=texts)
            items = sorted(resp.data, key=lambda d: d.index)
            if len(items) == len(texts):
                _batch_embeddings_supported = True
                return [item.embedding for item in items]
            # Beklenmedik sayıda vektör döndü: tek tek moda geç.
            _batch_embeddings_supported = False
        except Exception:
            _batch_embeddings_supported = False

    vectors: list[list[float]] = []
    for text in texts:
        resp = client.embeddings.create(model=model_id, input=text)
        vectors.append(resp.data[0].embedding)
    return vectors


def embed_text(client: OpenAI, model_id: str, text: str) -> list[float]:
    """Tek bir metin için embedding vektörü."""
    return embed_texts(client, model_id, [text])[0]


def shutdown() -> None:
    """Web servisini durdurur ve istemci önbelleğini temizler."""
    global _client

    manager = FoundryLocalManager.instance
    if manager is not None and manager.urls is not None:
        manager.stop_web_service()
    _client = None


@contextmanager
def foundry_model(alias: str, verbose: bool = True) -> Iterator[tuple[OpenAI, str]]:
    """(client, model_id) veren, çıkışta web servisini kapatan context manager."""
    try:
        yield get_model_client(alias, verbose=verbose)
    finally:
        shutdown()
