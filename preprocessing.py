import re

from config import STOPWORDS_PATH

class Preprocessor:
    def __init__(self, stopwords_path=None):
        self.stopwords = set()
        if stopwords_path is None:
            stopwords_path = STOPWORDS_PATH
        self.load_stopwords(stopwords_path)

    def load_stopwords(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    word = line.strip().lower()
                    if word and not word.startswith("#"):
                        self.stopwords.add(word)
        except OSError:
            # file stopword tidak ada / tidak terbaca: lanjut tanpa stopword
            self.stopwords = set()

    def tokenize(self, text):
        if not text:
            return []
        # case folding dan tokenisasi sederhana: angka/punksi dipisahkan
        text = text.lower()
        # sisakan hanya huruf/angka/spasi (tanda baca & simbol dipisahkan)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens

    def preprocess(self, text):
        tokens = self.tokenize(text)
        res = []
        for t in tokens:
            if t in self.stopwords:
                continue
            # hapus angka murni
            if t.isdigit():
                continue
            res.append(t)
        return " ".join(res)

    def preprocess_tokens(self, text):
        s = self.preprocess(text)
        return s.split() if s else []
