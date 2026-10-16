import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_loader import Document
from preprocessing import Preprocessor
from search_engine import SearchEngine


def make_docs(n=10):
    docs = []
    for i in range(n):
        docs.append(Document(
            f"t_{i:06d}",
            f"Judul dokumen {i} tentang teknologi",
            f"konten python belajar pemrograman komputer nomor {i}",
            "t",
        ))
    return docs


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.index_dir = os.path.join(self.tmp, "index")
        self.engine = SearchEngine(Preprocessor(), index_dir=self.index_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_index_and_search(self):
        self.engine.build_index(make_docs(5))
        res = self.engine.search("python", topk=5)
        self.assertTrue(len(res) >= 1)
        self.assertIn("score", res[0])
        self.assertGreaterEqual(res[0]["score"], 0.0)
        self.assertEqual(res[0]["rank"], 1)

    def test_whoosh_index_files_created(self):
        self.engine.build_index(make_docs(3))
        files = os.listdir(self.index_dir)
        self.assertTrue(any(f.endswith(".seg") for f in files), files)
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "manifest.json")))

    def test_whoosh_finds_only_relevant_candidates(self):
        docs = make_docs(5)
        docs.append(Document("x_000099", "Resep masakan", "cara membuat rendang enak", "x"))
        self.engine.build_index(docs)
        res = self.engine.search("rendang", topk=5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["doc_id"], "x_000099")

    def test_empty_query(self):
        self.engine.build_index(make_docs(3))
        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("    "), [])

    def test_query_only_stopwords(self):
        self.engine.build_index(make_docs(3))
        self.assertEqual(self.engine.search("dan atau yang"), [])

    def test_query_not_in_vocabulary(self):
        self.engine.build_index(make_docs(3))
        self.assertEqual(self.engine.search("zzzqqq"), [])

    def test_max_five_results(self):
        self.engine.build_index(make_docs(20))
        res = self.engine.search("python", topk=5)
        self.assertLessEqual(len(res), 5)
        # nilai topk lebih besar pun dibatasi 5
        res2 = self.engine.search("python", topk=99)
        self.assertLessEqual(len(res2), 5)

    def test_ranking_sorted_descending(self):
        docs = make_docs(20)
        # dokumen yang lebih mirip: judul+isi penuh frasa query
        docs[7] = Document("t_000007", "python python python",
                           "python python python python", "t")
        self.engine.build_index(docs)
        res = self.engine.search("python", topk=5)
        scores = [r["score"] for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res[0]["doc_id"], "t_000007")

    def test_search_before_ready_raises(self):
        with self.assertRaises(RuntimeError):
            self.engine.search("python")

    def test_bow_cache_roundtrip(self):
        docs = make_docs(4)
        self.engine.build_index(docs)
        self.engine.save_bow_cache()
        engine2 = SearchEngine(Preprocessor(), index_dir=self.index_dir)
        manifest = engine2.open_index()
        self.assertIsNotNone(manifest)
        self.assertTrue(engine2.load_bow_cache(expected_docs=4))
        res = engine2.search("python", topk=5)
        self.assertTrue(len(res) >= 1)

    def test_cache_mismatch_rejected(self):
        self.engine.build_index(make_docs(4))
        self.engine.save_bow_cache()
        engine2 = SearchEngine(Preprocessor(), index_dir=self.index_dir)
        engine2.open_index()
        self.assertFalse(engine2.load_bow_cache(expected_docs=999))

    def test_index_rebuild_no_duplicate(self):
        docs = make_docs(5)
        self.engine.build_index(docs)
        self.engine.build_index(docs)  # rebuild ulang
        from whoosh import index as whoosh_index
        ix = whoosh_index.open_dir(self.index_dir, indexname="docs")
        self.assertEqual(ix.doc_count(), 5)

    def test_missing_manifest_means_invalid(self):
        self.engine.build_index(make_docs(3))
        os.remove(os.path.join(self.index_dir, "manifest.json"))
        self.assertFalse(self.engine.is_index_valid())
        # masih bisa dibangun ulang
        self.engine.build_index(make_docs(3))
        self.assertTrue(self.engine.is_index_valid(expected_docs=3))

    def test_corrupt_index_handled(self):
        # index rusak: file .seg berisi sampah
        os.makedirs(self.index_dir, exist_ok=True)
        with open(os.path.join(self.index_dir, "docs_rusak.seg"), "w") as f:
            f.write("bukan file whoosh")
        self.assertFalse(self.engine.is_index_valid())
        self.engine.build_index(make_docs(2))  # harus tetap bisa
        self.assertTrue(self.engine.is_index_valid(expected_docs=2))

    def test_wrong_doc_count_invalid(self):
        self.engine.build_index(make_docs(4))
        self.assertFalse(self.engine.is_index_valid(expected_docs=5))
        self.assertTrue(self.engine.is_index_valid(expected_docs=4))

    def test_scores_are_real_cosine(self):
        from sklearn.metrics.pairwise import cosine_similarity as cs
        docs = make_docs(5)
        self.engine.build_index(docs)
        res = self.engine.search("python", topk=5)
        qvec = self.engine.vectorizer.transform([
            self.engine.preprocessor.preprocess("python")])
        expected = cs(qvec, self.engine.doc_matrix).flatten()
        best = max(expected)
        self.assertAlmostEqual(res[0]["score"], float(best), places=6)

    def test_title_field_searchable(self):
        docs = [
            Document("a_000001", "Kucing Oren Lucu", "hewan peliharaan rumahan", "a"),
            Document("a_000002", "Tentang Kopi", "minuman panas berkafein", "a"),
        ]
        self.engine.build_index(docs)
        res = self.engine.search("kucing", topk=5)
        self.assertTrue(any(r["doc_id"] == "a_000001" for r in res))


if __name__ == "__main__":
    unittest.main()
