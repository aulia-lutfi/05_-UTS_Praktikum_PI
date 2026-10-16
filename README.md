# Sistem Temu Kembali Informasi (Information Retrieval System)

Proyek UTS Praktikum Penelusuran Informasi: sistem temu kembali informasi berbasis
Command-Line Interface (CLI) yang mengindeks lima dataset teks berbahasa Indonesia
(dua kumpulan tesis/disertasi, dua kumpulan berita, satu kumpulan artikel populer),
lalu mencari dan meranking dokumen paling mirip dengan query pengguna.

## Informasi Kelompok
Nomor Kelompok: 05
 Anggota Kelompok:
  1. Aulia Lutfi
  2. Ilham
  3. Fadlur

## 1. Deskripsi Singkat

Sistem melakukan:

1. **Text preprocessing**: case folding, tokenization, dan stopword removal bahasa
   Indonesia (daftar stopword lokal, tanpa stemming/lemmatization).
2. **Indexing & searching dengan Whoosh** (library resmi Whoosh 2.7.4).
3. **Representasi Bag of Words (BoW)** memakai `CountVectorizer` dari scikit-learn.
4. **Ranking dengan cosine similarity** dari scikit-learn.
5. Menampilkan **maksimal 5 dokumen** dengan skor tertinggi.

## 2. Fitur Utama

- Membaca 5 dataset CSV sekaligus dan menyeragamkan strukturnya menjadi
  `doc_id`, `title`, `content`, `source`.
- `doc_id` unik untuk seluruh korpus (format `<sumber>_<nomor urut global 6 digit>`,
  mis. `etd_usk_000001`, `kompas_020001`).
- Index Whoosh disimpan di folder `index/` dan dapat dibangun ulang
  (dibersihkan lebih dulu, sehingga tidak menumpuk duplikat).
- Representasi BoW (vectorizer + matriks dokumen) dibangun **satu kali** saat
  menu *Load & Index Dataset*, tidak dihitung ulang setiap query.
- Cache `index/bow.pkl` supaya sesi berikutnya tidak perlu memproses ulang teks
  (hanya dimuat bila jumlah dokumennya cocok dengan manifest index).
- CLI interaktif dengan penanganan input tidak valid, file hilang, query kosong,
  dan error runtime yang informatif.

## 3. Struktur Folder

```
uts_pi/
├── main.py                 # CLI: menu, alur aplikasi
├── config.py               # path, nama sumber, pengaturan
├── dataset_loader.py       # pembaca 5 dataset CSV -> struktur dokumen seragam
├── preprocessing.py        # case folding, tokenization, stopword removal
├── search_engine.py        # Whoosh (index & kandidat), BoW, cosine similarity
├── stopwords_id.txt        # daftar stopword bahasa Indonesia (lokal)
├── requirements.txt        # dependensi eksternal
├── README.md
├── dataset/
│   ├── etd_usk.csv         # tesis/disertasi Universitas Syiah Kuala
│   ├── etd_ugm.csv         # tesis/disertasi Universitas Gadjah Mada
│   ├── kompas.csv          # berita nasional (Kompas)
│   ├── tempo.csv           # berita dan opini (Tempo)
│   └── mojok.csv           # artikel populer dan satir (Mojok)
├── index/                  # hasil indexing Whoosh + cache BoW (dibuat otomatis)
└── tests/
    ├── test_preprocessing.py
    ├── test_loader.py
    ├── test_search_engine.py
    ├── test_cli.py
    ├── test_integration_subset.py     # integrasi subset dataset asli (1.500 dokumen)
    └── integration_real_dataset.py    # integrasi penuh 49.484 dokumen (skrip)
```

## 4. Versi Python dan Dependensi

- Python 3.11 (diuji pada Python 3.11.0b4; minimal 3.8).
- Dependensi eksternal (lihat `requirements.txt`):
  - `scikit-learn` (CountVectorizer, cosine_similarity)
  - `pandas` (membaca CSV)
  - `Whoosh` (indexing dan pencarian)

Tidak ada library lain yang dipakai. Daftar stopword disimpan lokal di
`stopwords_id.txt`, tidak ada unduhan model/resource saat program berjalan.

## 5. Membuat Virtual Environment

Windows (PowerShell / CMD):

```bat
cd uts_pi
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
cd uts_pi
python3 -m venv .venv
source .venv/bin/activate
```

## 6. Memasang Dependensi

```bash
pip install -r requirements.txt
```

Isi `requirements.txt` (versi yang diuji; bisa dilonggarkan bila perlu):

```
scikit-learn==1.8.0
pandas==2.3.3
Whoosh==2.7.4
```

## 7. Lokasi dan Format Dataset

Lima file CSV wajib berada di folder `dataset/` dengan nama berikut:

| Sumber    | File                | Keterangan                        |
|-----------|---------------------|-----------------------------------|
| etd_usk   | `dataset/etd_usk.csv` | tesis/disertasi Universitas Syiah Kuala |
| etd_ugm   | `dataset/etd_ugm.csv` | tesis/disertasi Universitas Gadjah Mada |
| kompas    | `dataset/kompas.csv`  | berita nasional                   |
| tempo     | `dataset/tempo.csv`   | berita dan opini                  |
| mojok     | `dataset/mojok.csv`   | artikel populer dan satir         |

Format: CSV dengan header `judul,konten` (dua kolom teks, encoding UTF-8).

Jumlah dokumen pada dataset yang tersedia:

| Sumber   | Jumlah dokumen |
|----------|---------------:|
| etd_usk  | 10.000         |
| etd_ugm  | 10.000         |
| kompas   | 10.000         |
| tempo    | 10.000         |
| mojok    | 9.484          |
| **Total**| **49.484**     |

Catatan: pembaca bersifat toleran. Baris dengan judul dan isi kosong dilewati,
judul kosong diberi fallback `[Tidak ada judul] - <sumber>`, isi kosong memakai
judul sebagai teks yang dapat dicari, dan file yang hilang dicatat sebagai gagal
tanpa menghentikan pemrosesan sumber lain.

Hasil pemeriksaan dataset yang tersedia (disimpan di git sebagai file CSV):

- Encoding terbaca penuh sebagai UTF-8 oleh pandas.
- Tidak ada nilai kosong (NaN) pada kolom `judul` maupun `konten`.
- Tidak ada baris duplikat penuh; ada beberapa duplikat judul (etd_ugm: 3,
  kompas: 1) — tidak masalah karena `doc_id` tetap unik per baris.
- Ada dokumen dengan isi sangat pendek: etd_usk 62 dokumen (isi hanya
  "ABSTRAK") dan etd_ugm 94 dokumen (isi hanya "-"). Semuanya tetap dibaca
  (0 dilewati) dan tetap dapat ditemukan melalui judulnya.
- Median panjang isi: etd_usk 1,6 KB, etd_ugm 1,7 KB, kompas 2,2 KB,
  tempo 2,7 KB, mojok 5,1 KB.

## 8. Cara Menjalankan Program

Dari folder proyek (`uts_pi`), jalankan:

```bash
python main.py
```

Semua path di `config.py` dihitung dari lokasi file `config.py`, sehingga program
dapat dijalankan dari working directory mana pun. Untuk dataset yang berbeda,
ubah `DATASET_DIR`/`DATA_SOURCES` di `config.py`.

## 9. Cara Menggunakan Setiap Menu

```
=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================
Pilih menu:
```

- **[1] Load & Index Dataset**
  1. Membaca kelima CSV dan menampilkan jumlah `dibaca`, `dilewati`, `gagal`
     per sumber beserta totalnya.
  2. Mempersiapkan teks (preprocessing) dan membangun index Whoosh.
  3. Membangun BoW: `CountVectorizer` di-`fit` pada seluruh dokumen, lalu
     menghasilkan matriks dokumen (sparse).
  4. Menampilkan ringkasan: jumlah term vocabulary dan bentuk matriks dokumen.
  Jika index Whoosh di `index/` sudah ada dan jumlah dokumennya cocok, index
  **tidak** dibangun ulang; cache BoW dimuat bila tersedia.

- **[2] Search Query**
  1. Meminta input query; query kosong ditolak dengan pesan jelas.
  2. Query dipreprocess dengan fungsi yang sama seperti dokumen
     (case folding → tokenization → stopword removal).
  3. Whoosh mengambil kandidat dokumen (maksimal `WHOOSH_MAX_CANDIDATES`, default
     1000, pencarian pada field `title` dan `content` yang sudah dipreprocess).
  4. Kandidat diranking ulang dengan **cosine similarity** antara vektor query
     dan vektor dokumen (BoW) menggunakan vocabulary yang sama.
  5. Maksimal 5 hasil teratas ditampilkan: peringkat, skor cosine, sumber
     dataset, `doc_id`, dan judul.

- **[3] Exit** — keluar dari program.

### Membangun ulang index dan cache

Index Whoosh dan cache BoW berada di folder `index/`:

- `index/manifest.json` — jumlah dokumen index (dipakai untuk validasi).
- `index/bow.pkl` — hasil pickle dari `CountVectorizer`, matriks dokumen
  (sparse), dan metadata ringkas dokumen (`doc_id`, judul, sumber — tanpa teks
  isi).
- file `docs_*` — segmen index Whoosh (berisi teks yang sudah dipreprocess).

Cara membangun ulang semuanya: hapus folder `index/` lalu jalankan program dan
pilih menu [1]. Jika hanya cache BoW yang dihapus, menu [1] akan membangun ulang
BoW dari dokumen namun memakai kembali index Whoosh yang sudah ada. Cache BoW
diabaikan otomatis bila jumlah dokumennya tidak cocok dengan manifest.

### Pengaturan di `config.py`

| Pengaturan              | Default          | Arti |
|-------------------------|------------------|------|
| `DATASET_DIR`           | `./dataset`      | folder dataset (dihitung dari lokasi `config.py`) |
| `INDEX_DIR`             | `./index`        | folder index Whoosh + cache BoW |
| `DATA_SOURCES`          | 5 pasangan (nama, file) | daftar sumber dataset |
| `WHOOSH_MAX_CANDIDATES` | `1000`           | jumlah maksimal kandidat yang diambil Whoosh sebelum diranking cosine |
| `MAX_DOCS_PER_SOURCE`   | `None`           | batas dokumen per sumber; `None` = semua. Set mis. `2000` untuk demo cepat |

## 10. Contoh Query dan Format Hasil

Contoh nyata (hasil run pada dataset asli, disederhanakan tampilannya untuk
README; di program judul ditampilkan penuh):

```
=== INFORMATION RETRIEVAL SYSTEM ===
...
Pilih menu: 2
Masukkan query: kecerdasan buatan
Top 5 hasil untuk query: "kecerdasan buatan"
----------------------------------------------------------------------------------------------------
[1] skor=0.6036 | sumber=etd_ugm  | id=etd_ugm_011465
    INTERNSHIP REPORT: PT KPMG SIDDHARTA ADVISORY "HOW ARTIFICIA...
[2] skor=0.5938 | sumber=etd_ugm  | id=etd_ugm_018747
    THE ARTIFICIAL INTELLIGENCE PATENT TOWARDS ITS INVENTIONS AN...
[3] skor=0.5260 | sumber=mojok    | id=mojok_046786
    Kecerdasan Buatan Bisa Nggak Cerdas Lagi kalau Gantikan PNS ...
[4] skor=0.4193 | sumber=etd_ugm  | id=etd_ugm_013528
    Korelasi Antara Kreativitas Pengalaman Seni Komunitas SAGA d...
[5] skor=0.3576 | sumber=etd_usk  | id=etd_usk_007354
    PERBANDINGAN HASIL TANGKAPAN UMPAN PANCING GURITA SEMI BUATA...
----------------------------------------------------------------------------------------------------
```

Format setiap baris hasil: `[peringkat] skor=<cosine> | sumber=<dataset> |
id=<doc_id>` diikuti judul dokumen (dipotong bila lebih dari 70 karakter).

Interpretasi skor: skor adalah cosine similarity antara vektor query dan vektor
dokumen (bernilai 0–1), bukan ukuran relevansi sempurna. Skor tinggi berarti
kedua vektor berbagi banyak term dengan bobot serupa pada representasi BoW.

Contoh query lain yang bisa didemokan: `harga saham teknologi`,
`korupsi anggaran daerah`, `pembelajaran matematika siswa`, `lagu anime`.

## 11. Cara Menjalankan Pengujian

Pengujian unit dan integrasi ringan (tanpa library tambahan):

```bash
python -m unittest discover -s tests -v
```

Pengujian integrasi pada **dataset asli** (membangun index untuk seluruh 49.484
dokumen, menjalankan beberapa query, lalu menyimpan cache BoW). Jalankan sebagai
skrip karena memakan waktu beberapa menit:

```bash
python tests/integration_real_dataset.py
```

Hasil pengujian: seluruh pengujian `unittest` berstatus OK; hasil aktual
integrasi dataset asli dicatat pada bagian 12 di bawah.

## 12. Catatan Verifikasi Integrasi Dataset Asli

Dijalankan pada `python tests/integration_real_dataset.py` (Windows, Python 3.11.0b4,
scikit-learn 1.8.0, pandas 2.3.3, Whoosh 2.7.4). Output aktual:

```
[load] 49484 dokumen dalam 4.7s
   etd_usk  read=10000  skipped=0 failed=0
   etd_ugm  read=10000  skipped=0 failed=0
   kompas   read=10000  skipped=0 failed=0
   tempo    read=10000  skipped=0 failed=0
   mojok    read=9484   skipped=0 failed=0

[index+bow] 219.7s | BoW=(49484, 275785) nnz=7230201 vocab=275785
[cache] bow.pkl disimpan
```

Ringkasan:
- Kelima sumber berhasil dibaca: 49.484 dokumen, 0 dilewati, 0 gagal.
- Membangun index Whoosh + BoW: **219,7 detik** (~3,7 menit) pada run terisolasi.
  Pada run lain yang berjalan bersamaan dengan pengujian otomatis (CPU berbagi),
  langkah yang sama memerlukan 416,8 detik, jadi waktu build bergantung beban
  CPU/mesin. Total proses termasuk membaca dataset: ~225 detik (terisolasi).
- Vocabulary BoW: **275.785 term**; matriks dokumen 49.484 x 275.785 dengan
  7.230.201 elemen tak-nol.
- Ukuran folder `index/`: sekitar **320 MB** (236 MB segmen Whoosh + 98 MB
  `bow.pkl`). Hapus folder `index/` untuk membebaskan ruang bila tidak dipakai.

Waktu pencarian per query (setelah index siap) pada run terisolasi:

| Query                          | Hasil | Waktu   |
|--------------------------------|------:|--------:|
| `kecerdasan buatan`            | 5     | 0,248 s |
| `harga saham teknologi`        | 5     | 0,288 s |
| `korupsi anggaran daerah`      | 5     | 0,281 s |
| `pembelajaran matematika siswa`| 5     | 0,244 s |
| `lagu anime`                   | 5     | 0,231 s |
| `zzzzq tidakada`               | 0     | 0,131 s |
| `dan yang di` (hanya stopword) | 0     | 0,000 s |

(Saat CPU sibuk, waktu per query terukur 0,22–0,54 detik — tetap jauh di bawah
satu detik. Hasil/peringkat/skor identik di setiap run.)

Contoh hasil nyata untuk `pembelajaran matematika siswa`:

```
[1] skor=0.7600 | sumber=mojok    | id=mojok_046071
    Pembelajaran Matematika Seharusnya SESAT, Maksudnya Seru, Se...
[2] skor=0.6804 | sumber=etd_usk  | id=etd_usk_007219
    MENGIDENTIFIKASI SOCIOMATHEMATICAL NORM SISWA MELALUI PROJEK...
[3] skor=0.6181 | sumber=etd_usk  | id=etd_usk_006155
    HASIL BELAJAR MATEMATIKA DAN KETERLIBATAN SISWA MELALUI PENE...
```

Skor tersebut adalah cosine similarity nyata dari vektor BoW (diuji pula di
`tests/test_search_engine.py::test_scores_are_real_cosine`, yang membandingkan
skor hasil pencarian dengan `cosine_similarity` yang dihitung ulang).

Saat index sudah ada, menu [1] memakai kembali index dan memuat cache BoW:
total waktu ~10 detik (hanya membaca CSV dan memuat `bow.pkl`), tanpa
membangun ulang index.

Pengujian otomatis: `python -m unittest discover -s tests` — **50 tes, OK**
(termasuk pengujian integrasi subset 1.500 dokumen dari dataset asli).

## 13. Keterbatasan Implementasi

1. **Kandidat dari Whoosh.** Sesuai arsitektur yang dipilih, Whoosh dipakai untuk
   indexing dan sebagai penyaring kandidat (maksimal `WHOOSH_MAX_CANDIDATES`
   dokumen teratas menurut BM25). Cosine similarity meranking ulang kandidat
   tersebut. Konsekuensinya: dokumen yang tidak masuk daftar kandidat Whoosh
   tidak ikut diranking. Untuk query pendek hal ini biasanya tidak masalah, tetapi
   *recall* tidak dijamin penuh. Nilai `WHOOSH_MAX_CANDIDATES` dapat dinaikkan di
   `config.py` bila ingin kandidat lebih banyak.
2. **Tanpa stemming/lemmatization.** Bentuk kata yang berbeda (mis. "mengajar"
   vs "mengajarkan") dianggap term berbeda. Ini sesuai ketentuan bahwa stemming
   bersifat opsional dan menghindari library eksternal.
3. **Stopword removal sederhana.** Daftar stopword statis bahasa Indonesia di
   `stopwords_id.txt`; kata di luar daftar tidak dihapus. Angka murni dihapus.
4. **Bobot BoW = frekuensi mentah.** Tidak memakai TF-IDF atau pembobotan lain,
   sesuai permintaan penggunaan `CountVectorizer`. Term yang muncul berulang
   menaikkan kesamaan; karena cosine similarity menormalkan panjang vektor,
   dokumen panjang tidak otomatis menang.
5. **Preprocessing ulang setiap kali menu [1].** Cache hanya menyimpan hasil BoW;
   bila `index/bow.pkl` tidak ada atau tidak cocok, teks dokumen dipreprocess
   ulang (butuh waktu).
6. **Judul dan isi digabung sebagai satu teks** untuk BoW; judul tidak diberi
   bobot khusus.
7. Whoosh mengindeks teks yang **sudah dipreprocess** agar konsisten dengan
   vocabulary BoW dan query, sehingga pencarian BM25 Whoosh maupun cosine
   similarity bekerja pada representasi term yang sama.
8. **Isi dokumen tidak ditampilkan.** CLI hanya menampilkan judul, sumber, dan
   `doc_id`; teks isi dipakai untuk pencarian tetapi tidak di-print ke terminal.
