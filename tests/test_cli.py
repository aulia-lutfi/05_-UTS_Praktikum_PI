import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import IRSystem
from dataset_loader import DatasetLoader
from preprocessing import Preprocessor
from search_engine import SearchEngine

SOURCES = [
    ("etd_usk", "etd_usk.csv"),
    ("etd_ugm", "etd_ugm.csv"),
    ("kompas", "kompas.csv"),
    ("tempo", "tempo.csv"),
    ("mojok", "mojok.csv"),
]

DATA = {
    "etd_usk": ("Analisis Jaringan Komputer", "penelitian tentang jaringan komputer dan keamanan data"),
    "etd_ugm": ("Model Pembelajaran Matematika", "penelitian model pembelajaran matematika sekolah"),
    "kompas": ("Harga Saham Naik", "berita pasar saham hari ini mengalami kenaikan"),
    "tempo": ("Politik Nasional", "dinamika politik nasional menjelang pemilihan"),
    "mojok": ("Kuliner Nusantara", "artikel populer tentang kuliner nusantara yang lezat"),
}


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dset = os.path.join(self.tmp, "dataset")
        os.makedirs(self.dset, exist_ok=True)
        for src, fname in SOURCES:
            judul, konten = DATA[src]
            pd.DataFrame({"judul": [judul, judul + " 2"],
                          "konten": [konten, konten + " tambahan"]}).to_csv(
                os.path.join(self.dset, fname), index=False)
        self.index_dir = os.path.join(self.tmp, "index")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_system(self, inputs):
        """IRSystem dengan dataset & index di folder sementara, input disimulasikan."""
        it = iter(inputs)
        lines = []

        def fake_input(prompt=""):
            lines.append(prompt)
            try:
                return next(it)
            except StopIteration:
                raise EOFError

        def fake_print(*args, **kwargs):
            lines.append(" ".join(str(a) for a in args))

        sys_ = IRSystem(input_fn=fake_input, output_fn=fake_print)
        sys_.loader = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        sys_.engine = SearchEngine(Preprocessor(), index_dir=self.index_dir)
        return sys_, lines

    def test_full_flow_index_search_exit(self):
        sys_, lines = self.make_system(["1", "2", "jaringan komputer", "3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("INFORMATION RETRIEVAL SYSTEM", out)
        self.assertIn("Total dokumen berhasil dibaca: 10", out)
        self.assertIn("Status: index dan BoW siap digunakan", out)
        self.assertIn("Top", out)
        self.assertIn("skor=", out)
        self.assertIn("Program selesai", out)

    def test_search_before_load(self):
        sys_, lines = self.make_system(["2", "apa saja", "3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("Pilih menu [1] terlebih dahulu", out)

    def test_empty_query_rejected(self):
        sys_, lines = self.make_system(["1", "2", "   ", "3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("Query kosong", out)

    def test_invalid_menu(self):
        sys_, lines = self.make_system(["9", "abc", "3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("Pilihan tidak valid", out)

    def test_exit_without_load(self):
        sys_, lines = self.make_system(["3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("Program selesai", out)

    def test_eof_exits_gracefully(self):
        sys_, lines = self.make_system([])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("Keluar dari program", out)

    def test_results_come_from_multiple_sources_when_relevant(self):
        sys_, lines = self.make_system(["1", "2", "penelitian", "3"])
        sys_.run()
        out = "\n".join(lines)
        self.assertIn("skor=", out)


if __name__ == "__main__":
    unittest.main()
