"""Integrasi: build index nyata dari 5 dataset, lalu uji beberapa query.

Jalankan: python -m tests.test_integration   (atau sebagai skrip)
Dibuat sebagai skrip agar index hasil build bisa diverifikasi manual.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_loader import DatasetLoader
from preprocessing import Preprocessor
from search_engine import SearchEngine
import config


def main():
    t0 = time.time()
    loader = DatasetLoader()
    docs, stats = loader.load_all()
    t_load = time.time() - t0
    total = sum(s["read"] for s in stats.values())
    print(f"[load] {total} dokumen dalam {t_load:.1f}s")
    for src, st in stats.items():
        print(f"   {src:<8} read={st['read']:<6} skipped={st['skipped']} failed={st['failed']}")

    engine = SearchEngine(Preprocessor())
    t1 = time.time()
    engine.build_index(docs)
    t_index = time.time() - t1
    print(f"[index+bow] {t_index:.1f}s | smartcsr_BoW={engine.doc_matrix.shape} "
          f"nnz={engine.doc_matrix.nnz} vocab={len(engine.vectorizer.vocabulary_)}")

    engine.save_bow_cache()
    print("[cache] bow.pkl disimpan")

    queries = [
        "kecerdasan buatan",
        "harga saham teknologi",
        "korupsi anggaran daerah",
        "pembelajaran matematika siswa",
        "lagu anime",
        "zzzzq tidakada",
        "dan yang di",
    ]
    for q in queries:
        tq = time.time()
        res = engine.search(q, topk=5)
        dt = time.time() - tq
        print(f"\n[query] \"{q}\" -> {len(res)} hasil ({dt:.3f}s)")
        for r in res:
            print(f"   [{r['rank']}] {r['score']:.4f} | {r['source']:<8} | {r['doc_id']} | {r['title'][:60]}")

    print("\n[total]", f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
