import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
INDEX_DIR = os.path.join(BASE_DIR, "index")
STOPWORDS_PATH = os.path.join(BASE_DIR, "stopwords_id.txt")

DATA_SOURCES = [
    ("etd_usk", "etd_usk.csv"),
    ("etd_ugm", "etd_ugm.csv"),
    ("kompas", "kompas.csv"),
    ("tempo", "tempo.csv"),
    ("mojok", "mojok.csv"),
]

# Whoosh: max kandidat yang diambil sebelum cosine rerank
WHOOSH_MAX_CANDIDATES = 1000

# Kalau mau demo cepat: set ke misalnya 2000 untuk batasi per sumber; None=semua
MAX_DOCS_PER_SOURCE = None
