# Asal source dan artefak

Source dikurasi dari workspace eksperimen `kaggle-ntl-review-20261020`.
Arsip tersebut dipertahankan lokal, termasuk builder lama, log, receipt, dataset
unduhan, dan environment. Repo berisi kode yang dipakai ulang, snapshot notebook
pilihan, konfigurasi input, dan laporan kecil yang diperlukan untuk review.

`reports/provenance.json` memetakan file yang disalin ke path relatif arsip dan
checksum asalnya. File teks dinormalisasi ke UTF-8/LF agar konsisten saat
checkout lintas platform. Objek JSON notebook tetap sama: source sel, metadata,
outputs, dan execution count tidak diubah oleh normalisasi. Karena format
barisnya berubah dari arsip Windows, checksum byte snapshot repo dapat berbeda
dari checksum byte file arsip. `configs/notebooks.json` memakai checksum repo.

Template metadata Kaggle diganti menjadi `YOUR_KAGGLE_USERNAME/<slug>`, dan
`code_file` mengikuti nama snapshot yang lebih pendek. Daftar input, Docker,
pilihan GPU/Internet dan batas waktu dipertahankan. Metadata persiapan Llama
ditulis sebagai template CPU/Internet ON yang sesuai tujuan notebook; template
ini bukan dump respons API dari run persiapan lama.

Kode CPU di `src/ntl` mempertahankan implementasi aslinya. Pengujian MRF dan
Hungarian dipindahkan dari regression suite terdahulu, dengan import package dan
lokasi hasil benchmark disesuaikan. Tes yang bergantung pada arsip mesin lama
tidak dimasukkan; snapshot dan metadata kini diperiksa oleh `ntl validate`.

Snapshot Kaggle sengaja tetap mandiri dengan source runtime tertanam. Library
CPU adalah salinan yang dapat diimpor; perubahan di library harus ditinjau dan
diintegrasikan secara sengaja ke notebook baru. Validator menolak snapshot
yang berubah tanpa pembaruan checksum. Menjalankan validasi tidak mengeksekusi
source notebook atau memuat bobot LLM.

Tidak ada bobot model, data kompetisi, raw generation pool, signed download URL,
atau kredensial yang dimasukkan dalam daftar salin. `.gitignore` memisahkan
storage lokal dari source. Credits dan referensi upstream ada di `NOTICE.md`
dan tetap tercantum dalam notebook aslinya.
