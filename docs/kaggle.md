# Menjalankan snapshot di Kaggle

Daftar profil dapat dilihat dengan `python scripts/project.py list`.
Ekspor dilakukan dari checkout:

```powershell
python scripts/project.py validate
python scripts/project.py export svc_002 --owner adityaakbar
```

Folder `artifacts/kaggle/svc_002/` berisi snapshot dan metadata siap unggah.
Perintah ekspor tidak mengakses akun atau mengirim submission. Metadata di Git
memakai placeholder username; `--owner` menggantinya hanya pada hasil ekspor.

## Autentikasi dan push manual

CLI opsional:

```powershell
python -m pip install -e ".[dev,kaggle]"
kaggle kernels push -p artifacts/kaggle/svc_002 -t 32400
```

Parameter `-t 32400` menetapkan batas sembilan jam secara eksplisit. Pada CLI
2.2.4 yang dipakai proyek, `kernels_push` membaca timeout dari argumen CLI;
nilai `session_timeout_seconds` di metadata saja tidak mengisi request tersebut.
`machine_shape` tetap dibaca dari metadata.

Simpan autentikasi di `~/.kaggle/kaggle.json` atau environment `KAGGLE_USERNAME`
dan `KAGGLE_KEY`. Proyek tidak membaca path Downloads khusus komputer lama dan
tidak memerlukan kredensial untuk validasi, pengujian CPU, atau ekspor lokal.

Untuk menyimpan snapshot melalui UI, unggah `.ipynb`, pasang seluruh input dalam
metadata, lalu pilih T4 x2 dan Internet OFF. Tidak ada job yang dijalankan oleh
proses perapian repo ini.

## Input dan persiapan

Dataset/model/kernel input tercantum per profil di `configs/kaggle/`. Beberapa
input cache dan base Llama adalah output notebook persiapan di akun asal.
Jika memakai akun lain, buat output persiapan sendiri dan sesuaikan
`kernel_sources` pada metadata ekspor. Mengganti `--owner` tidak menyalin input
atau memberikan akses ke output privat akun lain.

- `prepare_train_cache`: T4 x2, Internet OFF; menghasilkan sepuluh fitur train,
  tujuh Gaussian dan manifest parity. Tidak membuat submission kompetisi.
- `prepare_llama`: CPU, Internet ON; mengunduh base publik
  `unsloth/Meta-Llama-3.1-8B`, revision
  `e9a141a2091ea561b96483212645a2a05e6f99fc`, dan memverifikasi empat shard.
  Setelah tersedia sebagai input, inferensi hybrid berjalan offline.
- `diagnostic_72`: inference penuh dengan 72 teks publik. Notebook mengubah nama
  output prediksi menjadi `diagnostic-predictions.parquet`; jangan submit profil
  ini ke kompetisi.

Cache train memerlukan package versions yang sesuai manifest. Image Docker,
GPU, daftar input dan batas waktu dipertahankan pada konfigurasi snapshot.
Instalasi CPU lokal menggunakan dependency ranges di `pyproject.toml`; jangan
menyamakan environment CPU itu dengan environment cache Kaggle yang dipin.

## Submission kompetisi

Setelah versi notebook selesai dan `submission.parquet` tervalidasi, submit versi
kode tersebut melalui Kaggle agar dijalankan ulang pada hidden test. Mengunggah
Parquet tiga contoh publik tidak mereproduksi submission kode. Ref yang
diterima belum membuktikan hidden run berhasil; periksa skor dan pesan error.

Repo ini menyimpan snapshot dan utilitas lokal. Controller otomatis lama beserta
receipt, lock, dan log tetap berada dalam arsip workspace lokal, terpisah dari
source yang akan dipush.
