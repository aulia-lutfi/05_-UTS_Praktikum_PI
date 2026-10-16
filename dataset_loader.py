import os
import pandas as pd

from config import DATA_SOURCES, DATASET_DIR, MAX_DOCS_PER_SOURCE

class Document:
    def __init__(self, doc_id, title, content, source):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.source = source

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
        }

class DatasetLoader:
    def __init__(self, max_per_source=MAX_DOCS_PER_SOURCE, dataset_dir=DATASET_DIR, data_sources=DATA_SOURCES):
        self.max_per_source = max_per_source
        self.dataset_dir = dataset_dir
        self.data_sources = data_sources
        self.documents = []
        self.stats = {}

    def load_all(self):
        self.documents.clear()
        self.stats.clear()
        idx_global = 0
        for source, fname in self.data_sources:
            path = os.path.join(self.dataset_dir, fname)
            stat = {"file": fname, "read": 0, "skipped": 0, "failed": 0, "path": path}
            if not os.path.exists(path):
                stat["failed"] = stat.get("failed", 0) + 1
                stat["note"] = "file not found"
                self.stats[source] = stat
                continue
            try:
                df = pd.read_csv(path)
                if len(df) == 0:
                    stat["skipped"] = len(df)
                    self.stats[source] = stat
                    continue
                # kolom pertama = judul, kolom kedua = konten
                tcol = df.columns[0]
                ccol = df.columns[1] if len(df.columns) >= 2 else df.columns[0]
                n = len(df)
                if self.max_per_source is not None and isinstance(self.max_per_source, int):
                    n_take = min(n, self.max_per_source)
                else:
                    n_take = n
                for i in range(n_take):
                    try:
                        row = df.iloc[i]
                        raw_title = row.get(tcol)
                        raw_content = row.get(ccol)
                        title = "" if pd.isna(raw_title) else str(raw_title).strip()
                        content = "" if pd.isna(raw_content) else str(raw_content).strip()
                        # skip jika kedua kolom kosong
                        if title == "" and content == "":
                            stat["skipped"] += 1
                            continue
                        if title == "":
                            title = f"[Tidak ada judul] - {source}"
                        if content == "" or content.lower() in ("nan", "none"):
                            content = title  # fallback agar tetap bisa dicari
                        idx_global += 1
                        doc_id = f"{source}_{idx_global:06d}"
                        self.documents.append(Document(doc_id, title, content, source))
                        stat["read"] += 1
                    except Exception:
                        stat["failed"] += 1
            except Exception as e:
                stat["failed"] = 1
                stat["error"] = str(e)
            self.stats[source] = stat
        return self.documents, self.stats
