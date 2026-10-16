import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import Preprocessor

ID_STOPWORDS = {"dan", "yang", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah", "ini", "itu", "tidak", "atau"}


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.p = Preprocessor()

    def test_case_folding(self):
        res = self.p.preprocess_tokens("Halo DUNIA Dunia")
        self.assertEqual(res, ["halo", "dunia", "dunia"])

    def test_tokenization(self):
        self.assertEqual(self.p.preprocess_tokens("Belajar Pemrograman Python"),
                         ["belajar", "pemrograman", "python"])

    def test_punctuation_split(self):
        res = self.p.preprocess_tokens("Data, AI! (sangat) penting...")
        self.assertIn("data", res)
        self.assertIn("penting", res)
        self.assertNotIn(",", res)
        self.assertNotIn("penting...", res)

    def test_stopword_removal(self):
        res = self.p.preprocess_tokens("dan ini adalah contoh yang dimiliki")
        for sw in ["dan", "ini", "adalah", "yang"]:
            self.assertNotIn(sw, res)
        self.assertIn("contoh", res)
        self.assertIn("dimiliki", res)

    def test_stopwords_actually_loaded(self):
        self.assertTrue(ID_STOPWORDS.issubset(self.p.stopwords))

    def test_numbers_removed(self):
        res = self.p.preprocess_tokens("tahun 2025 ada 10 dokumen")
        self.assertNotIn("2025", res)
        self.assertNotIn("10", res)

    def test_empty_and_none_safe(self):
        self.assertEqual(self.p.preprocess(""), "")
        self.assertEqual(self.p.preprocess(None), "")
        self.assertEqual(self.p.preprocess_tokens(""), [])
        self.assertEqual(self.p.preprocess_tokens("   "), [])

    def test_unicode_and_symbols_safe(self):
        text = "Kebijakan “politik” — 100% aman? ✔ café naïve"
        res = self.p.preprocess_tokens(text)
        self.assertIsInstance(res, list)
        self.assertIn("kebijakan", res)

    def test_document_and_query_use_same_function(self):
        # konsistensi: preprocess dokumen dan query memberi token yang sama
        self.assertEqual(self.p.preprocess_tokens("JARINGAN Komputer!"),
                         self.p.preprocess_tokens("jaringan komputer"))

    def test_missing_stopword_file_is_safe(self):
        p = Preprocessor(stopwords_path="tidak_ada_file_ini.txt")
        self.assertEqual(p.stopwords, set())
        self.assertEqual(p.preprocess_tokens("belajar python"), ["belajar", "python"])


if __name__ == "__main__":
    unittest.main()
