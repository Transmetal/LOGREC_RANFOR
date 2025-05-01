# Prediksi Harga Cryptocurrency (BTC-USD) dengan Model Klasik ML

Repositori ini berisi kode Python dan notebook Jupyter yang menyertai **Bab 8** dari buku Anda (atau sebagai proyek mandiri) yang mendemonstrasikan pembangunan dan evaluasi model Machine Learning (ML) klasik untuk memprediksi harga Bitcoin (BTC-USD). Ini mencakup langkah-langkah mulai dari pengambilan data, *feature engineering*, pelatihan model, evaluasi, hingga pembuatan dashboard interaktif sederhana.

**Catatan:** Kode ini ditujukan untuk tujuan edukasi dan demonstrasi. Ini **bukan** merupakan nasihat finansial. Pasar cryptocurrency sangat volatil, dan performa model di masa lalu tidak menjamin hasil di masa depan. Selalu lakukan riset Anda sendiri (DYOR) dan gunakan manajemen risiko yang tepat.

## Ringkasan Konten

Repositori ini mencakup implementasi untuk:

1.  **Pengambilan Data:** Mengunduh data harga historis BTC-USD dari Yahoo Finance menggunakan `yfinance`.
2.  **Feature Engineering:** Membuat fitur-fitur teknikal yang relevan (termasuk indikator dasar dan lanjutan menggunakan `pandas_ta`) serta fitur lag harga dan volume.
3.  **Preprocessing:** Membersihkan data (menangani NaN) dan melakukan scaling fitur menggunakan `MinMaxScaler`.
4.  **Pelatihan Model Klasik:**
    *   **Klasifikasi (Prediksi Arah):** Melatih model Logistic Regression dan Random Forest Classifier untuk memprediksi apakah harga akan Naik atau Turun/Sama.
    *   **Regresi (Prediksi Perubahan Harga):** Melatih model Linear Regression untuk memprediksi perubahan harga (Close_Diff) berikutnya.
5.  **Evaluasi Model:** Mengevaluasi model klasifikasi menggunakan Akurasi, AUC, Laporan Klasifikasi, dan Confusion Matrix. Mengevaluasi model regresi menggunakan RMSE pada harga absolut yang direkonstruksi.
6.  **Visualisasi:** Membuat plot Confusion Matrix, ROC Curve, dan plot backtest untuk model regresi.
7.  **Penyimpanan Model & Scaler:** Menyimpan model yang telah dilatih dan objek scaler menggunakan `joblib`.
8.  **Dashboard Interaktif:** Skrip Streamlit (`dashboard_classic_btc.py`) yang memuat model/scaler tersimpan, mengambil data terbaru, melakukan prediksi arah dan estimasi harga, serta menampilkan hasilnya secara visual.

## Struktur Repositori

├── LOGREC_RANFOR/
│ ├── DB.ipynb # Notebook Penjelasan Detail Kode
│ ├── OP.ipynb # Notebook Pelatihan & Evaluasi Utama 
│ ├── classic_models_clf_train.ipynb # Notebook Fokus Pelatihan Model Klasik 
│ ├── dashboard_classic_btc.py # Skrip Aplikasi Dashboard Streamlit
│ ├── DB.html # (Opsional) Ekspor HTML dari DB.ipynb
│ ├── OP.html # (Opsional) Ekspor HTML dari OP.ipynb
│ ├── DB.pdf # (Opsional) Ekspor PDF dari DB.ipynb
│ ├── OP.pdf # (Opsional) Ekspor PDF dari OP.ipynb
│ ├── logreg_clf_BTC-USD_advTA_model.pkl # Model Logistic Regression tersimpan
│ ├── rf_clf_BTC-USD_advTA_model.pkl # Model Random Forest tersimpan
│ ├── linreg_reg_BTC-USD_advTA_model.pkl # Model Linear Regression tersimpan
│ ├── scaler_X_classic_combined_BTC-USD_advTA.pkl # Scaler untuk fitur X tersimpan
│ └── scaler_y_classic_reg_BTC-USD_advTA.pkl # Scaler untuk target Y (regresi) tersimpan
└── README.md # File ini

## Model yang Digunakan

*   **Klasifikasi:** Logistic Regression, Random Forest Classifier (dari `scikit-learn`)
*   **Regresi:** Linear Regression (dari `scikit-learn`)

## Fitur yang Digunakan (Input Model)

Model-model ini dilatih menggunakan 16 fitur berikut (sesuai definisi `FEATURE_COLUMNS_INPUT` dalam kode):

'BBB_20_2.0', 'CMF_20', 'Close', 'Close_Diff_lag1',
'Close_Diff_lag2', 'Close_Diff_lag3', 'MACD_12_26_9',
'OBV', 'RSI_14', 'SMA_20', 'STOCHd_14_3_3', 'STOCHk_14_3_3',
'Volume', 'Volume_lag1', 'Volume_lag2', 'Volume_lag3'


## Memulai

Untuk menjalankan kode dan dashboard ini, ikuti langkah-langkah berikut:

### Prasyarat

*   **Python:** Versi 3.11.7 sangat direkomendasikan. Versi 3.8+ mungkin juga kompatibel.
*   **Conda (Direkomendasikan):** Untuk manajemen environment yang mudah. Anda bisa menginstal [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (versi ringan) atau [Anaconda](https://www.anaconda.com/products/distribution).
*   **Git:** Untuk mengkloning repositori.

### Instalasi (Menggunakan Conda - Direkomendasikan)

1.  **Kloning Repositori:**
    ```bash
    git clone https://github.com/Transmetal/LOGREC_RANFOR.git
    cd LOGREC_RANFOR
    ```

2.  **Buat File Environment (`environment.yml`):**
    Buat file bernama `environment.yml` di root direktori proyek Anda dengan konten berikut (versi library disesuaikan berdasarkan impor Anda, mungkin perlu penyesuaian minor):
    ```yaml
    name: crypto_classic_ml # Anda bisa ganti nama environment ini
    channels:
      - defaults
      - conda-forge # Diperlukan untuk yfinance & pandas_ta
    dependencies:
      - python=3.11 # Sesuaikan jika versi Python Anda berbeda
      - pip
      - numpy
      - pandas
      - yfinance
      - matplotlib
      - scikit-learn
      - joblib
      - pandas-ta
      - streamlit
      - ipykernel # Untuk menggunakan env ini di Jupyter
    ```

3.  **Buat dan Aktifkan Environment Conda:**
    Buka terminal atau Anaconda Prompt di direktori proyek Anda, lalu jalankan:
    ```bash
    conda env create -f environment.yml
    conda activate crypto_classic_ml # Gunakan nama environment dari file yml
    ```
    Proses ini akan mengunduh dan menginstal semua library yang diperlukan dalam environment terisolasi.

### Instalasi (Menggunakan Pip - Alternatif)

1.  **Kloning Repositori:** (Sama seperti langkah 1 Conda)
2.  **Buat File Requirements (`requirements.txt`):**
    Buat file bernama `requirements.txt` dengan konten:
    ```
    numpy
    pandas
    yfinance
    matplotlib
    scikit-learn
    joblib
    pandas-ta
    streamlit
    ipykernel
    ```
3.  **Buat dan Aktifkan Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Linux/macOS
    # atau
    venv\Scripts\activate    # Di Windows
    ```
4.  **Instal Library:**
    ```bash
    pip install -r requirements.txt
    ```

### Menjalankan Notebook Pelatihan/Penjelasan

1.  Pastikan environment Anda (conda atau venv) sudah aktif.
2.  (Opsional, jika menggunakan venv/conda env baru) Daftarkan kernel ke Jupyter:
    ```bash
    python -m ipykernel install --user --name=crypto_classic_ml --display-name "Python (crypto_classic_ml)"
    ```
3.  Mulai Jupyter Notebook atau Jupyter Lab:
    ```bash
    jupyter notebook
    # atau
    jupyter lab
    ```
4.  Buka browser Anda dan navigasi ke URL yang diberikan.
5.  Buka file notebook (misalnya `OP.ipynb` atau `classic_models_clf_train.ipynb`) dari folder `notebooks/`.
6.  Jalankan sel-sel kode sesuai urutan. Notebook ini akan mengambil data, melatih model, mengevaluasi, dan menyimpan file `.pkl`.

### Menjalankan Dashboard Prediksi

1.  Pastikan environment Anda (conda atau venv) sudah aktif.
2.  Pastikan file model dan scaler (`.pkl`) sudah ada di direktori yang sama dengan skrip dashboard (atau sesuaikan path dalam `dashboard_classic_btc.py`). File-file ini seharusnya dihasilkan saat Anda menjalankan notebook pelatihan.
3.  Jalankan aplikasi Streamlit dari terminal di direktori proyek Anda:
    ```bash
    streamlit run dashboard_classic_btc.py
    ```
4.  Aplikasi web interaktif akan terbuka di browser Anda secara otomatis. Anda dapat melihat data terbaru, hasil prediksi arah dari LR dan RF, estimasi harga dari LinReg, serta grafik historis.

## Data

*   Data harga historis **BTC-USD** timeframe harian diambil dari **Yahoo Finance** menggunakan library `yfinance`.
*   Periode data untuk pelatihan model dalam notebook adalah dari **1 Januari 2017 hingga 1 Januari 2024**.
*   Dashboard akan secara otomatis mengambil data terbaru yang diperlukan saat tombol "Muat Data & Prediksi" diklik.

## Kontak

Hubungi saya untuk kode yang lain vilencearg@gmail.com
