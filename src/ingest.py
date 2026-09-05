"""Doküman ingestion hattı: parçala -> embed -> SQLite'a yaz.

data/documents/ altındaki her belgeyi okur, başlık/bölümlere göre parçalar,
her chunk için embedding üretir (Foundry Local'ın OpenAI-uyumlu uç noktası +
çözümlenmiş embedding model id'si ile) ve db.py aracılığıyla SQLite'a yazar.

Her çalıştırmada tablo sıfırlanır. Retrieval / LLM cevap üretimi burada YOK.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe src/ingest.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# src/ klasörünü içe aktarma yoluna ekle (config.py ve db.py yan komşular).
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
import db
import foundry

# Bir bölüm bu karakter sayısını aşarsa paragraflara bölünür.
MAX_CHARS = 1200

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


# --------------------------------------------------------------------------- #
# Parçalama (chunking)
# --------------------------------------------------------------------------- #
def _split_long(body: str) -> list[str]:
    """Uzun bir metni paragraf (boş satır) sınırında parçalara böler."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > MAX_CHARS:
            parts.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        parts.append(buf.strip())
    return parts or [body.strip()]


def chunk_document(path: Path) -> list[dict[str, str]]:
    """Bir belgeyi (source, title, content) chunk'larına ayırır.

    - Belge başlığı: ilk seviye-1 (#) başlık, yoksa dosya adı.
    - Her seviye>=2 (##) bölüm bir chunk olur.
    - Çok uzun bölümler paragraflara bölünür.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc_title = path.stem
    sections: list[dict] = []  # {"level", "heading", "body": [lines]}
    current: dict | None = None

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 1 and current is None and not sections:
                # İlk seviye-1 başlık = belge başlığı
                doc_title = heading
                continue
            if current is not None:
                sections.append(current)
            current = {"level": level, "heading": heading, "body": []}
        elif current is not None:
            current["body"].append(line)
        # seviye-1 başlıktan önceki/başlıksız satırlar yok sayılır
    if current is not None:
        sections.append(current)

    chunks: list[dict[str, str]] = []

    # Hiç alt bölüm yoksa: tüm gövdeyi paragraflara göre parçala.
    if not sections:
        body = "\n".join(lines).strip()
        for i, part in enumerate(_split_long(body)):
            chunks.append(
                {
                    "source": path.name,
                    "title": doc_title if i == 0 else f"{doc_title} ({i + 1})",
                    "content": part,
                }
            )
        return chunks

    for sec in sections:
        heading = sec["heading"]
        body = "\n".join(sec["body"]).strip()
        # Embedding bağlamı için başlığı içeriğe dahil et.
        full = f"{doc_title} — {heading}\n{body}".strip()
        if len(full) <= MAX_CHARS:
            chunks.append({"source": path.name, "title": heading, "content": full})
        else:
            for i, part in enumerate(_split_long(body)):
                content = f"{doc_title} — {heading}\n{part}".strip()
                title = heading if i == 0 else f"{heading} ({i + 1})"
                chunks.append({"source": path.name, "title": title, "content": content})

    return chunks


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    docs = sorted(config.DOCS_DIR.glob("*.md")) + sorted(config.DOCS_DIR.glob("*.txt"))
    docs = sorted(docs, key=lambda p: p.name)
    if not docs:
        raise SystemExit(f"Belge bulunamadı: {config.DOCS_DIR}")

    # Tabloyu sıfırla.
    db.init_db(config.DB_PATH)

    client, model_id = foundry.get_model_client(config.EMBED_MODEL_ALIAS)

    total_chunks = 0
    embedding_dim: int | None = None
    sample: dict | None = None
    sample_dim: int | None = None

    print("\nBelgeler işleniyor...")
    for path in docs:
        chunks = chunk_document(path)
        # Qwen3-Embedding model kartı: BELGE tarafına instruction/prefix EKLENMEZ,
        # instruction yalnızca sorgu tarafında kullanılır (bkz. retrieve.py).
        vectors = foundry.embed_texts(client, model_id, [ch["content"] for ch in chunks])
        for ch, vector in zip(chunks, vectors):
            if embedding_dim is None:
                embedding_dim = len(vector)
            db.insert_chunk(
                config.DB_PATH,
                source=ch["source"],
                title=ch["title"],
                content=ch["content"],
                embedding=vector,
            )
            if sample is None:
                sample = ch
                sample_dim = len(vector)
        total_chunks += len(chunks)
        print(f"  {path.name}: {len(chunks)} chunk")

    foundry.shutdown()

    # --- Özet ---
    print("\n=== INGESTION ÖZETİ ===")
    print(f"İşlenen belge sayısı : {len(docs)}")
    print(f"Üretilen chunk sayısı: {total_chunks}")
    print(f"Embedding boyutu     : {embedding_dim}")
    print(f"Veritabanı           : {config.DB_PATH}")

    if sample is not None:
        print("\n--- Örnek chunk ---")
        print(f"source : {sample['source']}")
        print(f"title  : {sample['title']}")
        print(f"content:\n{sample['content']}")
        print(f"embedding uzunluğu: {sample_dim} (vektörün kendisi yazdırılmadı)")


if __name__ == "__main__":
    main()
