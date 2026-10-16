import os
import json
import pickle

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import MultifieldParser, OrGroup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import Preprocessor
import config as _config
from config import WHOOSH_MAX_CANDIDATES

MANIFEST_NAME = "manifest.json"
BOW_CACHE_NAME = "bow.pkl"


class SearchEngine:
    def __init__(self, preprocessor=None, index_dir=None):
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self.index_dir = index_dir if index_dir is not None else _config.INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self.ix = None
        self.schema = Schema(
            doc_id=ID(stored=True, unique=True),
            source=STORED,
            title=TEXT(stored=True),
            content=TEXT(stored=True),
        )
        self.vectorizer = None
        self.doc_matrix = None
        self.docs = []  # list of dicts

    # ---------------------------------------------------------------- index

    def build_index(self, documents):
        """Bangun ulang (overwrite) index Whoosh dan representasi BoW.

        Teks dokumen dipreprocess SEKALI saja, hasilnya dipakai untuk Whoosh
        maupun BoW.
        """
        self.docs = [d.to_dict() if hasattr(d, "to_dict") else d for d in documents]
        corpus = [
            self.preprocessor.preprocess(f"{d.get('title', '')} {d.get('content', '')}")
            for d in self.docs
        ]

        self.ix = index.create_in(self.index_dir, self.schema, indexname="docs")
        writer = self.ix.writer()
        for d, text in zip(self.docs, corpus):
            writer.add_document(
                doc_id=d["doc_id"],
                source=d["source"],
                title=d["title"],
                content=text,
            )
        writer.commit()

        self._fit_bow(corpus)
        self._write_manifest(len(self.docs))

    def _write_manifest(self, doc_count):
        manifest = {
            "doc_count": doc_count,
            "index_dir": self.index_dir,
        }
        path = os.path.join(self.index_dir, MANIFEST_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def _read_manifest(self):
        path = os.path.join(self.index_dir, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def open_index(self):
        """Buka index yang sudah ada dan jadikan aktif. Return manifest atau None."""
        try:
            if not index.exists_in(self.index_dir, indexname="docs"):
                return None
            ix = index.open_dir(self.index_dir, indexname="docs")
        except Exception:
            return None
        manifest = self._read_manifest()
        if manifest is None:
            return None
        self.ix = ix
        return manifest

    def is_index_valid(self, expected_docs=None):
        """Validasi index yang ada: ada index, ada manifest, jumlah dokumen cocok."""
        manifest = self.open_index()
        if manifest is None:
            return False
        if expected_docs is not None and manifest.get("doc_count") != expected_docs:
            return False
        return True

    # ---------------------------------------------------------------- BoW

    def build_bow(self, documents):
        """Bangun BoW tanpa membuat/mengubah index Whoosh."""
        self.docs = [d.to_dict() if hasattr(d, "to_dict") else d for d in documents]
        corpus = [
            self.preprocessor.preprocess(f"{d.get('title', '')} {d.get('content', '')}")
            for d in self.docs
        ]
        self._fit_bow(corpus)

    def _fit_bow(self, corpus):
        # corpus sudah bertoken (dipreprocess oleh Preprocessor), jadi analyzer
        # cukup memisahkan spasi. Ini menjaga vocabulary BoW identik dengan
        # hasil preprocessing dan membuat transformasi query konsisten
        # (tidak memakai token_pattern bawaan yang aturannya berbeda).
        self.vectorizer = CountVectorizer(analyzer=str.split)
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def save_bow_cache(self):
        """Simpan vectorizer + matriks dokumen + metadata ringkas ke pickle agar
        sesi berikutnya tidak perlu memproses ulang teks.

        Teks isi dokumen tidak disimpan (tidak diperlukan untuk pencarian);
        yang disimpan hanya doc_id, judul, dan sumber.
        """
        path = os.path.join(self.index_dir, BOW_CACHE_NAME)
        docs_meta = [
            {"doc_id": d.get("doc_id"), "title": d.get("title"), "source": d.get("source")}
            for d in self.docs
        ]
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer,
                 "doc_matrix": self.doc_matrix,
                 "docs": docs_meta},
                f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bow_cache(self, expected_docs=None):
        """Muat cache BoW. Return True jika berhasil dan cocok dengan index aktif."""
        path = os.path.join(self.index_dir, BOW_CACHE_NAME)
        if not os.path.exists(path):
            return False
        if self.ix is None:
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return False
        self.vectorizer = data.get("vectorizer")
        self.doc_matrix = data.get("doc_matrix")
        self.docs = data.get("docs", [])
        if self.vectorizer is None or self.doc_matrix is None:
            return False
        if expected_docs is not None and len(self.docs) != expected_docs:
            return False
        if self.doc_matrix.shape[0] != len(self.docs):
            return False
        return True

    # ---------------------------------------------------------------- cari

    def search(self, query_str, topk=5, max_cand=WHOOSH_MAX_CANDIDATES):
        if self.ix is None:
            raise RuntimeError(
                "Index Whoosh belum siap. Pilih menu [1] Load & Index Dataset terlebih dahulu.")
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError(
                "Representasi BoW belum siap. Pilih menu [1] Load & Index Dataset terlebih dahulu.")

        query_str = (query_str or "").strip()
        if query_str == "":
            return []

        # query dipreprocess dengan fungsi yang sama seperti dokumen
        qproc = self.preprocessor.preprocess(query_str)
        if qproc == "":
            return []

        topk = max(1, min(int(topk), 5))  # maksimal 5 sesuai spesifikasi

        # 1) Whoosh: ambil kandidat dokumen
        parser = MultifieldParser(["title", "content"], self.ix.schema, group=OrGroup)
        with self.ix.searcher() as searcher:
            try:
                q = parser.parse(qproc)
                results = searcher.search(q, limit=max_cand)
            except Exception:
                return []
            candidates = [r["doc_id"] for r in results]

        if not candidates:
            return []

        # 2) Cosine similarity: ranking kandidat dengan BoW
        id2idx = {d["doc_id"]: i for i, d in enumerate(self.docs)}
        cand_idxs = [id2idx[c] for c in candidates if c in id2idx]
        if not cand_idxs:
            return []

        qvec = self.vectorizer.transform([qproc])
        if qvec.nnz == 0:
            # query tidak memiliki kosakata dalam vocabulary BoW
            return []

        D = self.doc_matrix[cand_idxs]
        sims = cosine_similarity(qvec, D).flatten()

        items = list(zip(cand_idxs, sims))
        items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, s in items[:topk]:
            if s <= 0:
                continue  # hanya tampilkan dokumen yang benar-benar berbagi kosakata
            d = self.docs[idx]
            results.append({
                "rank": len(results) + 1,
                "doc_id": d["doc_id"],
                "title": d["title"],
                "source": d["source"],
                "score": float(s),
            })
        return results
