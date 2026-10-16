from dataset_loader import DatasetLoader
from preprocessing import Preprocessor
from search_engine import SearchEngine
import config

MENU_TEXT = """=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
===================================="""


class IRSystem:
    def __init__(self, input_fn=input, output_fn=print):
        self.input = input_fn
        self.out = output_fn
        self.preprocessor = Preprocessor()
        self.loader = DatasetLoader()
        self.engine = SearchEngine(self.preprocessor)
        self.docs = []
        self.stats = {}
        self.ready = False

    # ------------------------------------------------------------ menu 1

    def load_and_index(self, force=False):
        self.out("[1] Membaca kelima dataset...")
        self.docs, self.stats = self.loader.load_all()
        total = sum(s.get("read", 0) for s in self.stats.values())
        for src, st in self.stats.items():
            note = f" ({st['file']})"
            if st.get("error"):
                note += f" ERROR: {st['error']}"
            if st.get("note"):
                note += f" [{st['note']}]"
            self.out(f"    - {src:<8} dibaca={st['read']:<6} dilewati={st['skipped']:<4} "
                     f"gagal={st['failed']:<3}{note}")
        self.out(f"    Total dokumen berhasil dibaca: {total}")

        if total == 0:
            self.out(
                "[!] Tidak ada dokumen yang berhasil dibaca. Index tidak dibangun.")
            self.ready = False
            return False

        # Cek index yang sudah ada: kalau valid dan tidak dipaksa, jangan bangun ulang.
        manifest = self.engine.open_index()
        if (not force) and manifest is not None and manifest.get("doc_count") == total:
            self.out(f"[1] Index Whoosh yang ada cocok ({total} dokumen). "
                     f"Index tidak dibangun ulang.")
            if self.engine.load_bow_cache(expected_docs=total):
                self.out("    Cache BoW dimuat (tanpa preprocessing ulang).")
            else:
                self.out("    Membangun representasi BoW dari dokumen...")
                self.engine.build_bow(self.docs)
                self.engine.save_bow_cache()
            self.ready = True
            self._print_summary()
            return True

        self.out(f"[1] Membangun index Whoosh + BoW untuk {total} dokumen "
                 f"(bisa beberapa menit, mohon tunggu)...")
        self.engine.build_index(self.docs)
        self.engine.save_bow_cache()
        self.out(f"    Index disimpan di: {config.INDEX_DIR}")
        self.ready = True
        self._print_summary()
        return True

    def _print_summary(self):
        vocab = len(self.engine.vectorizer.vocabulary_)
        self.out(f"    Vocabulary BoW     : {vocab} term")
        self.out(f"    Matriks dokumen    : {self.engine.doc_matrix.shape[0]} dokumen x "
                 f"{self.engine.doc_matrix.shape[1]} fitur")
        self.out("[1] Status: index dan BoW siap digunakan.")

    # ------------------------------------------------------------ menu 2

    def search_query(self, query):
        if not self.ready and self.engine.vectorizer is None:
            self.out("[!] Dataset belum dimuat dan index belum siap. "
                     "Pilih menu [1] terlebih dahulu.")
            return []
        query = (query or "").strip()
        if query == "":
            self.out("[!] Query kosong. Masukkan kata kunci yang ingin dicari.")
            return []
        results = self.engine.search(query, topk=5)
        if not results:
            self.out("[i] Tidak ada dokumen yang cocok dengan query tersebut.")
            return []
        self.out(f"Top {len(results)} hasil untuk query: \"{query}\"")
        self.out("-" * 100)
        for r in results:
            title = r["title"]
            if len(title) > 70:
                title = title[:67] + "..."
            self.out(f"[{r['rank']}] skor={r['score']:.4f} | sumber={r['source']:<8} | "
                     f"id={r['doc_id']}")
            self.out(f"    {title}")
        self.out("-" * 100)
        return results

    # ------------------------------------------------------------ loop CLI

    def run(self):
        while True:
            self.out("\n\n")
            self.out(MENU_TEXT)
            try:
                choice = self.input("Pilih menu: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.out("\nKeluar dari program.")
                break
            if choice == "1":
                try:
                    self.load_and_index()
                except Exception as e:
                    self.out(
                        f"[!] Gagal memuat/mengindeks dataset: {type(e).__name__}: {e}")
            elif choice == "2":
                try:
                    query = self.input("Masukkan query: ").strip()
                except (EOFError, KeyboardInterrupt):
                    self.out("\nKeluar dari program.")
                    break
                try:
                    self.search_query(query)
                except RuntimeError as e:
                    self.out(f"[!] {e}")
                except Exception as e:
                    self.out(
                        f"[!] Error saat pencarian: {type(e).__name__}: {e}")
            elif choice == "3":
                self.out("Terima kasih. Program selesai.")
                break
            else:
                self.out("[!] Pilihan tidak valid. Masukkan 1, 2, atau 3.")


def main():
    try:
        IRSystem().run()
    except KeyboardInterrupt:
        print("\nProgram dihentikan.")


if __name__ == "__main__":
    main()
