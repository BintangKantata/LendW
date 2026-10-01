# Loan Approval Assistant — Prototype Webapp

Webapp Flask satu halaman untuk memprediksi risiko gagal bayar (default risk)
menggunakan model dari `loan_default_risk_assistant.ipynb`.

## Isi folder

```
webapp/
├── app.py                    # Backend Flask
├── requirements.txt          # Daftar dependency Python
├── templates/
│   └── index.html            # Halaman UI (form + hasil prediksi)
└── loan_default_model.pkl    # Model hasil training (kamu yang taruh di sini)
```

## ⚠️ Catatan penting: Windows + model ini

Model ini (`loan_default_model.pkl`) dilatih di Google Colab (Linux) dan
memakai `xgboost==3.4.1`. Ada **bug yang sudah dikonfirmasi oleh tim XGBoost**
([github.com/dmlc/xgboost/issues/12459](https://github.com/dmlc/xgboost/issues/12459))
yang menyebabkan model XGBoost hasil training di Linux **gagal dibuka
langsung di Windows** lewat pickle (`XGBoostError: input stream corrupted`),
walau file-nya tidak corrupt dan semua versi library persis sama.

**Kalau kamu menjalankan ini di Windows asli (bukan WSL/Docker), kamu akan
kemungkinan besar mengalami error ini.** Dua cara aman untuk menghindarinya:

### Opsi A — Docker (disarankan, paling gampang kalau sudah ada Docker Desktop)

1. Taruh `loan_default_model.pkl` di folder ini (sejajar dengan `Dockerfile`).
2. Build image:
   ```
   docker build -t loan-assistant .
   ```
3. Jalankan container:
   ```
   docker run -p 5000:5000 loan-assistant
   ```
4. Buka `http://127.0.0.1:5000` di browser Windows kamu seperti biasa.

### Opsi B — WSL (Windows Subsystem for Linux)

1. Install WSL dengan distro Ubuntu (bukan `docker-desktop` WSL default):
   ```
   wsl --install -d Ubuntu
   ```
2. Buka terminal Ubuntu, masuk ke folder project (lewat `/mnt/c/...`), lalu
   jalankan langkah "Cara menjalankan" di bawah **di dalam WSL**, bukan di
   Command Prompt/PowerShell biasa.

---

## Cara menjalankan (Linux/macOS/WSL/Docker)

1. **Taruh file model** — pastikan `loan_default_model.pkl` (hasil download dari Colab)
   ada di folder yang sama dengan `app.py`.

2. **Buat virtual environment (disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependency**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server**
   ```bash
   python app.py
   ```
   Kamu akan melihat log seperti:
   ```
   Model dimuat: XGBoost (calibrated) (threshold=0.190)
   * Running on http://127.0.0.1:5000
   ```

5. **Buka browser** ke `http://127.0.0.1:5000` — isi form (Income, Loan Amount,
   Credit Score, Employment, Debt), klik **Predict Risk**, dan hasilnya
   (persentase risiko, label LOW/MEDIUM/HIGH RISK, main factors, assessment)
   akan muncul di panel kanan.

## Catatan versi library

`requirements.txt` mengunci versi scikit-learn ke **1.6.1** — ini harus sama
dengan versi yang dipakai saat training di notebook (Colab), karena objek
`joblib`/pickle sklearn (mis. `ColumnTransformer`, `CalibratedClassifierCV`)
tidak selalu backward/forward compatible antar versi mayor sklearn. Kalau kamu
training ulang dengan versi sklearn lain, sesuaikan juga versi di sini.

## Cara kerja singkat

- `app.py` meng-load `loan_default_model.pkl` sekali saat start (dict berisi
  `pipeline`, `threshold`, `model_name`).
- Endpoint `POST /predict` menerima 5 input dari form (Income, Loan Amount,
  Credit Score, Employment tahun, Debt), melengkapi kolom-kolom lain yang
  dibutuhkan pipeline (Age, NumCreditLines, InterestRate, LoanTerm, Education,
  dst) dengan nilai median/modus dari dataset training, lalu memanggil
  `pipeline.predict_proba()`.
- Hasil probabilitas dibandingkan dengan `threshold` yang tersimpan (bukan
  0.5) untuk menentukan status "flagged", dan dipetakan ke label
  LOW/MEDIUM/HIGH RISK untuk ditampilkan di UI.
- "Main factors" dihasilkan dari aturan sederhana (rule-based) yang
  membandingkan input dengan angka referensi dataset (credit score ≥ 700,
  income ≥ median, DTI < 30%, dst) — bukan dari feature importance model
  secara langsung. Ini cukup untuk prototype; untuk versi produksi,
  pertimbangkan mengganti dengan SHAP values (lihat notebook bagian
  "Langkah Selanjutnya").

## Troubleshooting

- **`FileNotFoundError: loan_default_model.pkl`** — pastikan file model ada
  persis di folder yang sama dengan `app.py`, dan namanya sama persis.
- **`InconsistentVersionWarning` saat load model** — biasanya tidak fatal,
  tapi kalau sampai error, pastikan `pip install -r requirements.txt`
  benar-benar terpasang (bukan versi sklearn lain yang sudah ter-install
  sebelumnya).
- **Model terlalu besar / lambat di-load** — wajar, karena
  `CalibratedClassifierCV` menyimpan beberapa salinan classifier untuk
  cross-validation kalibrasi. Loading pertama kali bisa makan waktu
  beberapa detik.
