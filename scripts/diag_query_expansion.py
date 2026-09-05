"""Ölçüm: sorgu genişletme (query expansion) retrieval'ı düzeltir mi, bozar mı?

Teşhis (scripts/diag_colors.py) şunu gösterdi: "renkler neler" gibi kısa, ürün
bağlamı olmayan sorgularda embedding gürültü üretiyor — tüm skorlar 0.38-0.44
arasında sıkışıyor ve doğru chunk top-3'e giremiyor. Aynı bilgi düzgün kurulmuş
"Hangi renkler var" sorgusuyla 1. sırada geliyor. Yani darboğaz belgede değil,
SORGU tarafında.

Fikir: embedding'den ÖNCE sorgunun başına sabit bir ürün bağlamı eklemek.

    (A) mevcut     : Instruct: {görev}\\nQuery:{sorgu}
    (B) genişletilmiş: Instruct: {görev}\\nQuery:{ÜRÜN BAĞLAMI}{sorgu}

Qwen3-Embedding model kartının önerdiği instruction prefix'i KORUNUR; ürün bağlamı
"Query:" yuvasının içine, yani kullanıcı metninin başına eklenir (instruction görev
tanımı içindir, kullanıcı metni değil).

ÖNEMLİ: Genişletme yalnızca RETRIEVAL içindir. Kullanıcıya gösterilen ve chat
modeline giden asıl soru DEĞİŞMEZ.

Bu script SALT OKUR — retrieve.py veya başka kalıcı dosya değiştirilmez; tüm
genişletme mantığı buradadır.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe scripts/diag_query_expansion.py
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

TOP_K = config.TOP_K

# Denenen sabit ürün bağlamı.
PRODUCT_CONTEXT = "KahveUsta 300 akıllı kahve makinesi hakkında soru: "


def format_plain(query: str) -> str:
    """(A) Mevcut davranış — retrieve.format_query ile birebir aynı."""
    return retrieve.format_query(query)


def format_expanded(query: str) -> str:
    """(B) Instruction prefix korunur, ürün bağlamı Query yuvasına eklenir."""
    return retrieve.format_query(f"{PRODUCT_CONTEXT}{query}")


STRATEGIES = [
    ("A: mevcut", format_plain),
    ("B: genişletilmiş", format_expanded),
]


# Hedef chunk tanımları: (etiket, predicate). predicate None ise hedef yok.
def color_target(ch: dict) -> bool:
    return ch["source"] == "01_teknik_ozellikler.md" and ch["title"] == "Renk Seçenekleri"


def garanti_target(ch: dict) -> bool:
    return ch["source"] == "06_garanti_destek_iletisim.md"


def kirec_target(ch: dict) -> bool:
    return ch["source"] == "04_temizlik_bakim.txt" and "kireç" in ch["title"].casefold()


def wifi_target(ch: dict) -> bool:
    return ch["source"] == "07_mobil_uygulama_wifi.md"


QUERIES = [
    ("renkler neler", "01 — Renk Seçenekleri", color_target, "şu an BAŞARISIZ"),
    ("renk seçenekleri neler", "01 — Renk Seçenekleri", color_target, "şu an BAŞARISIZ"),
    ("Hangi renkler var", "01 — Renk Seçenekleri", color_target, "şu an başarılı"),
    ("Garanti süresi ne kadar?", "06 — garanti belgesi", garanti_target, "bozulmamalı"),
    ("Kireç çözmeyi ne sıklıkla yapmalıyım?", "04 — Kireç Çözme", kirec_target, "bozulmamalı"),
    ("Wi-Fi'ye nasıl bağlanırım?", "07 — mobil/Wi-Fi", wifi_target, "bozulmamalı"),
    ("Bu makine su geçirmez mi?", None, None, "belgede YOK — skorlar yükselmemeli"),
]


def rank_all(chunks: list[dict], normalized: np.ndarray,
             client, model_id: str, formatted_query: str) -> np.ndarray:
    """Verilen (biçimlendirilmiş) sorgu için tüm chunk'ların kosinüs skorları."""
    vector = np.array(foundry.embed_text(client, model_id, formatted_query), dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return normalized @ vector


def target_position(scores: np.ndarray, chunks: list[dict], predicate) -> tuple[int, float, int]:
    """Hedefe uyan en iyi chunk'ın (sıra, skor, indeks) bilgisi."""
    order = np.argsort(-scores)
    for position, idx in enumerate(order, start=1):
        if predicate(chunks[idx]):
            return position, float(scores[idx]), int(idx)
    return -1, float("nan"), -1


def main() -> None:
    print("=== ÖLÇÜM: sorgu genişletme (query expansion) ===")
    print(f"Embed model : {config.EMBED_MODEL_ALIAS}")
    print(f"top_k       : {TOP_K}")
    print(f"Instruction : {retrieve.QUERY_TASK_DESCRIPTION}")
    print(f"Ürün bağlamı: {PRODUCT_CONTEXT!r}\n")

    client, model_id = foundry.get_model_client(config.EMBED_MODEL_ALIAS)
    chunks, normalized = retrieve.load_index()
    print(f"Toplam chunk: {len(chunks)}")

    summary: list[tuple] = []

    try:
        for query, target_label, predicate, note in QUERIES:
            print("\n" + "=" * 78)
            print(f"SORGU: {query!r}   [{note}]")
            print(f"Hedef: {target_label or '(yok)'}")
            print("=" * 78)

            per_strategy: dict[str, tuple] = {}

            for label, formatter in STRATEGIES:
                formatted = formatter(query)
                scores = rank_all(chunks, normalized, client, model_id, formatted)
                order = np.argsort(-scores)[:TOP_K]

                print(f"\n  [{label}]")
                print(f"  gönderilen sorgu: {formatted!r}")
                for rank, idx in enumerate(order, start=1):
                    ch = chunks[idx]
                    print(f"    {rank}. skor={scores[idx]:.4f} | {ch['source']} | {ch['title']}")

                if predicate is None:
                    top_score = float(scores[order[0]])
                    print(f"    >> en yüksek skor: {top_score:.4f} (hedef yok)")
                    per_strategy[label] = (None, None, top_score)
                else:
                    position, score, idx = target_position(scores, chunks, predicate)
                    in_top = any(predicate(chunks[i]) for i in order)
                    threshold = float(scores[order[TOP_K - 1]])
                    print(
                        f"    >> hedef: sıra {position}/{len(chunks)}, skor {score:.4f}; "
                        f"top-{TOP_K} eşiği {threshold:.4f}; "
                        f"top-{TOP_K} içinde: {'EVET' if in_top else 'HAYIR'}"
                    )
                    per_strategy[label] = (position, score, in_top)

            summary.append((query, target_label, predicate is None, per_strategy))
    finally:
        foundry.shutdown()

    # ------------------------------------------------------------------ özet
    print("\n\n" + "=" * 78)
    print("KARŞILAŞTIRMA ÖZETİ")
    print("=" * 78)
    labels = [s[0] for s in STRATEGIES]
    print(f"{'Sorgu':40s}{'A: sıra/skor':>20s}{'B: sıra/skor':>20s}{'  sonuç'}")
    print("-" * 96)

    for query, _target_label, no_target, per_strategy in summary:
        name = query if len(query) <= 38 else query[:35] + "..."
        if no_target:
            a_top = per_strategy[labels[0]][2]
            b_top = per_strategy[labels[1]][2]
            verdict = "B skoru YÜKSELDİ" if b_top > a_top + 0.02 else "skorlar benzer"
            print(f"{name:40s}{('en iyi %.4f' % a_top):>20s}"
                  f"{('en iyi %.4f' % b_top):>20s}  {verdict}")
            continue

        a_pos, a_score, a_in = per_strategy[labels[0]]
        b_pos, b_score, b_in = per_strategy[labels[1]]
        if a_in and b_in:
            verdict = "ikisi de OK"
        elif b_in and not a_in:
            verdict = "B DÜZELTTİ"
        elif a_in and not b_in:
            verdict = "B BOZDU"
        else:
            verdict = "ikisi de başarısız"
        print(f"{name:40s}{('%d / %.4f' % (a_pos, a_score)):>20s}"
              f"{('%d / %.4f' % (b_pos, b_score)):>20s}  {verdict}")

    print("-" * 96)
    print("sıra = hedef chunk'ın tüm sıralamadaki yeri (1 = en iyi)")


if __name__ == "__main__":
    main()
