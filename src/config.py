"""Merkezi yapılandırma.

Model alias'ları, yol ayarları ve RAG sabitleri tek yerde tutulur.
"""

from pathlib import Path

# Proje kökü: src/ klasörünün bir üstü.
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Model alias'ları (Foundry Local kataloğundaki adlar) ---
# qwen3-1.7b denendi ve RAG için yetersiz bulundu: bağlamdaki cevabı gözden
# kaçırıyor, kendini tekrar eden döngülere giriyor ve bağlamda olmayan bilgiyi
# uyduruyordu. qwen3-4b (düşünme modu açık) aynı test setinde 5/5 doğru.
CHAT_MODEL_ALIAS = "qwen3-4b"
EMBED_MODEL_ALIAS = "qwen3-embedding-0.6b"

# --- Yollar ---
DOCS_DIR = ROOT_DIR / "data" / "documents"
DB_PATH = ROOT_DIR / "knowledge.db"

# --- Retrieval sabitleri ---
TOP_K = 3
