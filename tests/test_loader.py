import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_loader import DatasetLoader

SOURCES = [
    ("etd_usk", "etd_usk.csv"),
    ("etd_ugm", "etd_ugm.csv"),
    ("kompas", "kompas.csv"),
    ("tempo", "tempo.csv"),
    ("mojok", "mojok.csv"),
]


def write_datasets(folder, rows_per_source=2):
    for src, fname in SOURCES:
        df = pd.DataFrame({
            "judul": [f"Judul {src} {i}" for i in range(rows_per_source)],
            "konten": [f"Konten {src} berisi informasi penting nomor {i}"
                       for i in range(rows_per_source)],
        })
        df.to_csv(os.path.join(folder, fname), index=False)


class TestLoaderSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dset = os.path.join(self.tmp, "dataset")
        os.makedirs(self.dset, exist_ok=True)
        write_datasets(self.dset, 2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_all_five_sources(self):
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, stats = dl.load_all()
        self.assertEqual(len(docs), 10)
        self.assertEqual({d.source for d in docs}, {s for s, _ in SOURCES})
        for src, _ in SOURCES:
            self.assertEqual(stats[src]["read"], 2)

    def test_doc_id_unique(self):
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, _ = dl.load_all()
        ids = [d.doc_id for d in docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_skips_empty_rows(self):
        path = os.path.join(self.dset, "kompas.csv")
        df = pd.DataFrame({"judul": ["", "Ada isi"], "konten": ["", "isi dokumen"]})
        df.to_csv(path, index=False)
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, stats = dl.load_all()
        self.assertEqual(stats["kompas"]["skipped"], 1)
        self.assertEqual(stats["kompas"]["read"], 1)

    def test_missing_file_handled(self):
        os.remove(os.path.join(self.dset, "tempo.csv"))
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, stats = dl.load_all()
        self.assertGreaterEqual(stats["tempo"]["failed"], 1)
        self.assertEqual(stats["tempo"]["read"], 0)
        # sumber lain tetap terbaca
        self.assertEqual(stats["kompas"]["read"], 2)

    def test_empty_dataset(self):
        for _, fname in SOURCES:
            path = os.path.join(self.dset, fname)
            pd.DataFrame({"judul": [], "konten": []}).to_csv(path, index=False)
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, stats = dl.load_all()
        self.assertEqual(len(docs), 0)

    def test_max_per_source(self):
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES,
                           max_per_source=1)
        docs, _ = dl.load_all()
        self.assertEqual(len(docs), 5)

    def test_title_fallback_for_empty_content(self):
        path = os.path.join(self.dset, "mojok.csv")
        pd.DataFrame({"judul": ["Hanya judul"], "konten": [""]}).to_csv(path, index=False)
        dl = DatasetLoader(dataset_dir=self.dset, data_sources=SOURCES)
        docs, _ = dl.load_all()
        mojok_docs = [d for d in docs if d.source == "mojok"]
        self.assertEqual(len(mojok_docs), 1)
        self.assertEqual(mojok_docs[0].content, "Hanya judul")


class TestLoaderRealDataset(unittest.TestCase):
    """Integrasi pada dataset asli yang tersedia di folder dataset/."""

    def test_real_dataset_counts_and_ids(self):
        import config
        if not os.path.exists(os.path.join(config.DATASET_DIR, "kompas.csv")):
            self.skipTest("dataset asli tidak tersedia")
        dl = DatasetLoader()
        docs, stats = dl.load_all()
        self.assertGreater(len(docs), 0)
        # kelima sumber terwakili
        for src, _ in SOURCES:
            self.assertGreater(stats[src]["read"], 0, f"sumber {src} kosong")
        ids = [d.doc_id for d in docs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(docs))


if __name__ == "__main__":
    unittest.main()
