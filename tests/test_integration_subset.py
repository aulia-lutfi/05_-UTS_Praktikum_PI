"""Pengujian integrasi pada dataset ASLI dengan subset kecil (cepat).

Memakai 300 dokumen per sumber (1.500 dokumen) dari folder `dataset/` yang asli,
lalu membangun index Whoosh + BoW dan menjalankan query. Untuk pengujian penuh
49.484 dokumen, jalankan `python tests/integration_real_dataset.py`.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from dataset_loader import DatasetLoader
from preprocessing import Preprocessor
from search_engine import SearchEngine

SOURCES = ["etd_usk", "etd_ugm", "kompas", "tempo", "mojok"]
SUBSET = 300


def datasets_available():
    return all(os.path.exists(os.path.join(config.DATASET_DIR, f"{s}.csv"))
               for s in SOURCES)


@unittest.skipUnless(datasets_available(), "dataset asli tidak tersedia")
class TestRealDatasetSubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ir_real_")
        cls.loader = DatasetLoader(max_per_source=SUBSET)
        cls.docs, cls.stats = cls.loader.load_all()
        cls.engine = SearchEngine(Preprocessor(), index_dir=cls.tmp)
        cls.engine.build_index(cls.docs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_five_sources_loaded(self):
        for src in SOURCES:
            self.assertEqual(self.stats[src]["read"], SUBSET, f"sumber {src}")
        self.assertEqual(len(self.docs), SUBSET * len(SOURCES))

    def test_doc_ids_unique(self):
        ids = [d.doc_id for d in self.docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_whoosh_index_has_all_docs(self):
        self.assertEqual(self.engine.ix.doc_count(), SUBSET * len(SOURCES))

    def test_bow_matrix_rows_match_docs(self):
        self.assertEqual(self.engine.doc_matrix.shape[0], len(self.docs))
        self.assertGreater(len(self.engine.vectorizer.vocabulary_), 0)

    def test_search_returns_at_most_five_sorted(self):
        for q in ["kecerdasan buatan", "harga saham", "pembelajaran matematika",
                  "lagu anime", "korupsi"]:
            res = self.engine.search(q, topk=5)
            self.assertLessEqual(len(res), 5, q)
            self.assertGreater(len(res), 0, q)
            scores = [r["score"] for r in res]
            self.assertEqual(scores, sorted(scores, reverse=True), q)
            for r in res:
                self.assertGreater(r["score"], 0.0)
                self.assertIn(r["source"], SOURCES)

    def test_unknown_query_returns_nothing(self):
        self.assertEqual(self.engine.search("zzzzqqqq"), [])

    def test_stopword_only_query_returns_nothing(self):
        self.assertEqual(self.engine.search("dan yang di untuk"), [])

    def test_index_manifest_and_reuse(self):
        manifest_path = os.path.join(self.tmp, "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["doc_count"], len(self.docs))
        self.engine.save_bow_cache()
        engine2 = SearchEngine(Preprocessor(), index_dir=self.tmp)
        self.assertTrue(engine2.is_index_valid(expected_docs=len(self.docs)))
        self.assertTrue(engine2.load_bow_cache(expected_docs=len(self.docs)))
        res = engine2.search("korupsi", topk=5)
        self.assertGreater(len(res), 0)


if __name__ == "__main__":
    unittest.main()
