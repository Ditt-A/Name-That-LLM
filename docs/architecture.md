# Alur inferensi

Baseline menggabungkan classifier utama dua tahap LoRA Nishant Qwen3-8B/14B
dengan ensemble pelengkap. Classifier utama tetap memiliki SVC gaya bahasa,
head plain/centered, dan Gaussian Qwen8B. Ensemble pelengkap memuat empat view
Gaussian serta readout/adapter lain dari sumber. Campuran sebelum ATLAS adalah
85% utama dan 15% pelengkap.

ATLAS memberikan bukti likelihood tambahan dengan bobot 10%. Pada baseline
0,608 dan varian SVC, ATLAS dijalankan untuk semua baris dengan batch 1. Setelah
itu, 100 iterasi Sinkhorn menyesuaikan massa kolom untuk test berukuran minimal
50 baris. Contoh publik tiga baris melewati balancing.

Soft MRF bekerja pada probabilitas akhir. Bloknya adalah irisan partisi prompt
TF-IDF dan GLM. Confidence adalah Jaccard kedua kelompok induk. Duplikasi label
di dalam blok membayar penalti `.40 * confidence`; label yang sama tetap boleh
dipilih lebih dari sekali. Solver slot label berulang mencari MAP tepat pada
clique kecil. Gate ukuran kelompok, probability loss, rank, energy gain, dan
configuration gap menentukan apakah seluruh blok menerima perubahan.

## SVC 2% yang direview

```text
B = .85 * classifier_utama + .15 * ensemble_pelengkap
C = .98 * B + .02 * SVC_gabungan
P = Sinkhorn(.90 * C + .10 * ATLAS)
label = SoftMRF(P)
```

SVC tambahan memakai empat view TF-IDF: karakter, kata, shape, dan function
words. Pool Nishant dan Dhruv digabung, dinormalisasi whitespace, dibersihkan
dari konflik label dan overlap official train. Arsip mencatat 39.542 teks
eksternal unik ditambah 2.699 official train. Urutan 18 kelas SVC diselaraskan
dengan nama label checkpoint sebelum pencampuran.

Revisi SVC terbaru mengikuti alur inline notebook 0,608. Sel ATLAS, balancing,
MRF dan output dipertahankan. Reader CSV SVC mempertahankan ID sebagai string,
termasuk nol di depan. Tidak ada model Llama atau adapter Dhruv tambahan pada
varian ini.

## Hybrid dan kalibrasi

Hybrid memakai `.75 baseline + .20 Dhruv + .05 SVC` sebelum ATLAS. Stack Dhruv
terdiri dari Qwen3-8B, Qwen3-14B, Llama3.1-8B dan Qwen2.5-7B full-data adapters.
ATLAS bersifat selektif pada hybrid. Varian hybrid MRF mengganti decoder
Hungarian; campuran dan routing hybrid dipertahankan.

Varian SVC terkalibrasi memfit satu temperatur pada skor official train yang
dihasilkan tiga SVC dengan fold prompt terpisah. Alias prompt yang diketahui
dihubungkan melalui teks identik yang dinormalisasi. Duplikat semantik yang
tidak diketahui masih mungkin tersisa. Temperatur diterapkan hanya pada SVC
tambahan, bukan pada output in-sample pretrained Qwen. NLL sesudah pemilihan
temperatur bukan estimasi independen akurasi ensemble.

## Cache dan contoh publik

Cache persiapan menyimpan sepuluh set fitur official train dan tujuh Gaussian
yang sudah difit. Manifest mengikat official train, recipe, checkpoint,
tokenizer, adapter/head, metadata base dan nama/ukuran shard. File fitur dan
parameter Gaussian diperiksa dengan checksum. Base multi-GB tidak dihash ulang
pada setiap inferensi. Versi package cache ada di `configs/cache-environment.json`.

Fitur test tetap dihitung oleh model. Cache lain yang tertanam pada beberapa
notebook hanya berlaku pada tiga ID publik dengan SHA teks lengkap yang cocok
persis. Prediksi hidden tidak disimpan. Diagnostik 72 teks memakai model nyata
dan jalur penuh; teksnya berasal dari official train publik sehingga bukan
validasi akurasi independen.
