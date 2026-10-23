# Name That LLM

Eksperimen untuk kompetisi [Name That LLM](https://www.kaggle.com/competitions/name-that-llm):
atribusi penulis teks memakai checkpoint Qwen/LoRA pretrained, Gaussian heads,
SVC, ATLAS, dan decoding kelompok dengan Soft MRF atau Hungarian.

Skor tertinggi yang terverifikasi dalam arsip proyek adalah **0,608** untuk
baseline dengan Soft MRF. Hybrid besar dan versi cached-nya mendapat **0,602**.
Notebook eksperimen baru disimpan bersama provenance; skor yang belum dicatat
ditandai sebagai belum diketahui. Lihat [catatan eksperimen](docs/experiments.md).

Tidak ada fine-tuning ulang Qwen/LoRA dalam eksperimen ini. SVC, readout linear,
dan Gaussian tetap dapat difit sesuai recipe notebook; cache train menghindari
forward official train berulang. Bobot/data diperoleh melalui input Kaggle dan
tidak disimpan di Git.

## Struktur

```text
configs/             Profil notebook, metadata input Kaggle, kontrak/checksum cache
docs/                Arsitektur, eksperimen, penggunaan Kaggle, dan provenance
notebooks/
  baselines/         Soft MRF 0,608 sebagai pembanding yang berhasil
  experiments/       SVC 2%, SVC terkalibrasi, hybrid MRF, dan penalti MRF 0,55
  preparation/       Persiapan fitur train/Gaussian dan base Llama offline
  diagnostics/       Uji jalur penuh pada 72 teks publik
  reference/         Hybrid cached 0,602 untuk perbandingan
src/ntl/             Fungsi ensemble, kalibrasi, cache, checkpoint, dan decoder CPU
tests/               Regresi algoritme yang dipindahkan dari workspace eksperimen
reports/             Hasil pemeriksaan kecil yang sengaja disimpan
scripts/             Entry point utilitas dari checkout
data/                Data lokal; diabaikan Git
artifacts/           Ekspor Kaggle dan hasil run lokal; diabaikan Git
```

Notebook adalah snapshot mandiri. Fungsi dalam `src/ntl` dapat diimpor untuk
eksplorasi CPU; mengedit library tidak otomatis menulis ulang snapshot notebook.
Checksum dan asal setiap snapshot tercatat di `configs/notebooks.json` dan
`reports/provenance.json`.

## Persiapan lokal

Python 3.11 atau lebih baru:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ntl list
.\.venv\Scripts\python -m ntl validate
.\.venv\Scripts\python -m pytest
```

Untuk menjalankan utilitas langsung dari checkout setelah dependensi tersedia:

```powershell
python scripts/project.py list
python scripts/project.py validate
```

Validasi memeriksa checksum, struktur notebook, kompilasi sel dan modul Python
tertanam, metadata, serta file yang tidak semestinya masuk repo. Perintah ini
berjalan offline dan tidak menjalankan model GPU. Pengujian lokal bukan bukti
akurasi atau durasi inferensi hidden test.

## Ekspor notebook ke Kaggle

```powershell
python scripts/project.py export svc_002 --owner adityaakbar
```

Hasilnya ada di `artifacts/kaggle/svc_002/`: notebook dan `kernel-metadata.json`
yang sudah memakai username tujuan. Ekspor hanya membuat berkas lokal.

Untuk penggunaan CLI Kaggle, autentikasi berada di environment atau
`~/.kaggle/kaggle.json`. Contoh environment kosong ada di `.env.example`.
Lihat [panduan Kaggle](docs/kaggle.md) untuk push, input yang perlu disiapkan,
dan batas sesi. Seluruh notebook inferensi memakai **T4 x2, Internet OFF**;
persiapan base Llama memakai **CPU, Internet ON**.

## Kode inti

```python
from ntl.postprocessing.soft_mrf import soft_constraint_decode

labels, report = soft_constraint_decode(probabilities, tfidf_groups, glm_groups)
```

Decoder mengembalikan label terpisah dan mempertahankan matriks probabilitas.
Penalti dan gate adalah heuristik eksperimen; penjelasan metode tersedia di
[arsitektur](docs/architecture.md). Kredit kode/notebook/model upstream ada di
[NOTICE.md](NOTICE.md) dan [provenance](docs/provenance.md).

Repo tujuan: [Ditt-A/Name-That-LLM](https://github.com/Ditt-A/Name-That-LLM).
