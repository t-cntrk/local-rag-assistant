"""Retrieval katmanı: sorguyu embed'le -> kosinüs benzerliği -> en iyi K chunk.

Asimetrik retrieval notu (Qwen3-Embedding resmî model kartı):
    Sorgu tarafında bir instruction/prefix kullanılması önerilir; belgeler için
    ise "No need to add instruction for retrieval documents" denir. Önerilen
    biçim:

        Instruct: {task_description}\\nQuery:{query}

    Model kartı ayrıca instruction'ın 1-5 puanlık iyileşme sağladığını ve çok
    dilli senaryolarda bile instruction'ın **İngilizce** yazılmasını önerir
    (eğitim sırasında kullanılan instruction'lar İngilizceydi). Bu yüzden
    sorgunun kendisi Türkçe kalır, sadece görev tanımı İngilizcedir.

    Belgeler ingest.py'de bilinçli olarak prefix'siz embed'lenir.

Sorgu genişletme:
    Instruction prefix'e ek olarak, sorgunun başına sabit bir ürün bağlamı
    eklenir (RETRIEVAL_QUERY_CONTEXT). Bu yalnızca retrieval embedding'ini
    etkiler; modele giden soru değişmez.

LLM ile cevap üretimi burada YOK — o adım rag.py'ye ait.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/ klasörünü içe aktarma yoluna ekle (config.py, db.py, foundry.py komşular).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import config
import db
import foundry

# Sorgu tarafı görev tanımı (model kartı önerisi gereği İngilizce).
QUERY_TASK_DESCRIPTION = (
    "Given a user question about a coffee machine's user manual, "
    "retrieve the manual passages that answer the question"
)

# Bellek içi indeks önbelleği: (metadata listesi, normalize edilmiş matris)
_index: tuple[list[dict], np.ndarray] | None = None


def format_query(query: str) -> str:
    """Sorguyu Qwen3-Embedding'in önerdiği instruction biçimine sarar."""
    return f"Instruct: {QUERY_TASK_DESCRIPTION}\nQuery:{query}"


# Retrieval sorgusuna eklenen sabit ürün bağlamı (sorgu genişletme).
#
# Neden: kısa ve eksik kurulmuş sorgular ("renkler neler") tek başına embed
# edildiğinde ayırt edici bir vektör üretmiyordu — tüm chunk skorları 0.38-0.44
# arasında sıkışıyor, sıralama gürültüye dönüyor ve doğru parça top-3'e
# giremiyordu. Sorguya ürün bağlamı eklemek aynı parçayı 15/40'tan 1/40'a
# çıkardı; düzgün kurulmuş sorgular ise 1. sırada kalmaya devam etti.
# Ölçümler: scripts/diag_query_expansion.py (retrieval) ve
# scripts/diag_expansion_hallucination.py (uçtan uca; uydurma güvencesi korundu).
#
# Bu önek YALNIZCA retrieval embedding'i içindir: chat modeline giden ve
# kullanıcıya gösterilen soru değişmez.
RETRIEVAL_QUERY_CONTEXT = "KahveUsta 300 akıllı kahve makinesi hakkında soru: "


def embed_query(query: str, verbose: bool = False) -> list[float]:
    """Sorgunun embedding vektörünü (1024 boyut) üretir.

    Sorgu, embedding'den önce ürün bağlamıyla genişletilir (bkz.
    RETRIEVAL_QUERY_CONTEXT) ve ardından Qwen3-Embedding'in instruction
    biçimine sarılır. Genişletme yalnızca hangi chunk'ların getirileceğini
    etkiler; asıl soru olduğu gibi kalır.

    Embedding modeli foundry.py üzerinden bir kez yüklenir; sonraki çağrılarda
    aynı web servisi ve model yeniden kullanılır.
    """
    client, model_id = foundry.get_model_client(config.EMBED_MODEL_ALIAS, verbose=verbose)
    retrieval_query = format_query(f"{RETRIEVAL_QUERY_CONTEXT}{query}")
    return foundry.embed_text(client, model_id, retrieval_query)


def load_index(force_reload: bool = False) -> tuple[list[dict], np.ndarray]:
    """Tüm chunk'ları DB'den okur ve L2-normalize edilmiş matrisi döndürür.

    Normalize edilmiş vektörlerde kosinüs benzerliği = nokta çarpımı.
    """
    global _index

    if _index is not None and not force_reload:
        return _index

    chunks = db.get_all_chunks(config.DB_PATH)
    if not chunks:
        raise SystemExit(
            f"Veritabanında chunk yok: {config.DB_PATH}\n"
            "Önce ingestion'ı çalıştırın: .venv/Scripts/python.exe src/ingest.py"
        )

    matrix = np.array([ch["embedding"] for ch in chunks], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # sıfır vektöre karşı koruma
    _index = (chunks, matrix / norms)
    return _index


def search(
    query: str,
    top_k: int = config.TOP_K,
    verbose: bool = False,
) -> list[tuple[float, str, str, str]]:
    """Sorguya en yakın top_k chunk'ı döndürür.

    Dönüş: [(score, source, title, content), ...] — skora göre azalan sırada.
    score, [-1, 1] aralığında kosinüs benzerliğidir.
    """
    chunks, normalized = load_index()

    query_vector = np.array(embed_query(query, verbose=verbose), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Sorgu embedding'i sıfır vektör döndü.")
    query_vector = query_vector / query_norm

    if query_vector.shape[0] != normalized.shape[1]:
        raise ValueError(
            f"Boyut uyuşmazlığı: sorgu {query_vector.shape[0]}, "
            f"veritabanı {normalized.shape[1]}. Ingestion'ı aynı modelle tekrar çalıştırın."
        )

    scores = normalized @ query_vector

    k = min(top_k, len(chunks))
    # argpartition ile en iyi k'yı ayır, sonra kendi içinde sırala.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        (float(scores[i]), chunks[i]["source"], chunks[i]["title"], chunks[i]["content"])
        for i in top_idx
    ]
