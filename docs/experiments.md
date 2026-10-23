# Catatan eksperimen

Tabel ini merupakan snapshot bukti yang tersedia saat proyek dirapikan pada
23 Oktober 2026. `Diterima` berarti API mengembalikan ref submission; status
tersebut tidak menyatakan bahwa hidden evaluation selesai tanpa error.

| Profil | Metode | Skor terverifikasi di arsip | Ref submission |
|---|---|---:|---:|
| Referensi upstream | [ATLAS reproduction](https://www.kaggle.com/code/taqiyudinadn/ntl-public-atlas-reproduction?scriptVersionId=360593727) | 0,606 | 57464631 |
| `soft_mrf_0608` | Baseline + Soft MRF .40 | 0,608 | 57508672 |
| `hybrid_cached_0602` | Hybrid cached + Hungarian | 0,602 | 57528628 |
| `svc_002` | Alur inline .608 + SVC 2% + cache train | Diterima; skor belum dicatat | 57550025 |
| `svc_calibrated` | SVC 2% dengan temperatur dari tiga fold prompt | Diterima; skor belum dicatat | 57549578 |
| `hybrid_mrf` | Hybrid cached, decoder Soft MRF | Diterima; skor belum dicatat | 57550453 |
| `mrf_055` | Baseline + penalti Soft MRF .55 | Diterima; skor belum dicatat | 57549315 |

Hybrid tanpa cache juga mendapat 0,602. Ada beberapa variabel yang berubah
dibanding notebook 0,608: komposisi ensemble, routing ATLAS, dan decoder.
Perbedaan skor tidak cukup untuk mengisolasi penyebab pada satu model.

Versi sebelumnya dari hybrid MRF dan SVC 2% lolos run publik tetapi gagal
hidden dengan pesan unhandled error umum. Versi kalibrasi dan MRF .55 pernah
gagal saat mounting input sebelum Python berjalan. Input Llama/adapter Dhruv
yang tidak digunakan dilepas dari varian SVC/MRF .55 sebelum retry. Penyebab
hidden error sebelumnya belum dibuktikan oleh traceback hidden.

SVC direview terhadap kode .608 dan cached .602: 24 fungsi utama dan 53 fungsi
pelengkap cocok secara logika, empat sel downstream dipertahankan, dan 24
pemeriksaan CPU tambahan lolos. Ukuran kasus mencakup 3, 49, 50, 72 dan 4.000
baris. Regresi CLI ID membuktikan nol di depan tidak hilang. Laporan ringkas ada
di `reports/svc-source-comparison.json` dan `reports/svc-cpu-validation.json`.

Diagnostik 72 teks sudah selesai dengan inference pretrained nyata pada jalur
baseline, SVC, ATLAS semua baris dan MRF. Lihat `reports/diagnostic-72.json`.
Ini memeriksa alur dan numerik; tidak membuktikan durasi hidden 4.000 baris atau
memberikan pengukuran akurasi yang independen dari data pelatihan checkpoint.
