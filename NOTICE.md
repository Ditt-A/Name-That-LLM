# Kredit dan sumber

Proyek mengadaptasi pekerjaan publik berikut. Kredit di dalam notebook sumber
dipertahankan; model, adapter, dataset, dan kode upstream tetap mengikuti
ketentuan sumber masing-masing. Perapian repo ini tidak menetapkan lisensi baru
untuk seluruh materi pihak ketiga.

- [Taqiyudin — public ATLAS reproduction](https://www.kaggle.com/code/taqiyudinadn/ntl-public-atlas-reproduction?scriptVersionId=360593727):
  baseline pembanding .606.
- [Joseph Adamski — Teach the Model + 10% ATLAS](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas):
  ATLAS dan sumber ensemble publik.
- [Two-Stage LoRA + Gaussian + Full Hungarian](https://www.kaggle.com/code/sangam21/two-stage-lora-gaussian-full-hungarian-0-604/notebook):
  salah satu sumber awal eksperimen.
- [Nishant Dahal — two-stage LoRA assets](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora):
  checkpoint dua tahap dan generation pool.
- [Lucas Boesen](https://www.kaggle.com/lucasboesen) dan kontributor upstream:
  pretrained encoders/readouts, Gaussian views, token-attention dan adapter.
- [Dhruv Tibarewal — nlm-v21](https://www.kaggle.com/code/dhruvtibarewal/nlm-v21):
  adapter full-data, pool tambahan, serta metode hybrid pembanding.
- [Vishak — v1-vishak](https://www.kaggle.com/code/vishakkbhat/v1-vishak):
  recipe SVC empat view yang diadaptasi dengan pool yang dapat diakses.
- [Basu, Bilenko & Mooney, KDD 2004](https://www.cs.utexas.edu/~ml/papers/semi-kdd-04.pdf):
  inspirasi constraint probabilistik. Decoder proyek bukan reproduksi HMRF-KMEANS.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html):
  referensi temperature scaling untuk eksperimen kalibrasi SVC.

Daftar source lengkap notebook dan sumber bobot/data dapat ditelusuri melalui
`configs/kaggle/`, metadata dalam notebook, dan `reports/provenance.json`.
