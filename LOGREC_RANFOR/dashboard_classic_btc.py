# --- START OF FILE dashboard_classic_combined.py (Tidak perlu diubah dari versi terakhir) ---

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression, LinearRegression # Impor LinearRegression
from sklearn.ensemble import RandomForestClassifier # Impor juga jika pakai RF Regressor
import joblib
import matplotlib.pyplot as plt
import os 

# Coba impor pandas_ta
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    st.error("ERROR: Library pandas_ta tidak ditemukan. Mohon install: pip install pandas_ta")
    st.stop() 
else:
    PANDAS_TA_AVAILABLE = True

# --- Konfigurasi Dasar ---
DEFAULT_TICKER = 'BTC-USD'
TIME_STEP_CONTEXT = 60 
BUFFER_DAYS = 55 # Sesuaikan buffer jika perlu

# --- Nama File Model & Scaler ---
LOGREG_MODEL_SAVE_PATH = f'logreg_clf_{DEFAULT_TICKER}_advTA_model.pkl'
RF_MODEL_SAVE_PATH = f'rf_clf_{DEFAULT_TICKER}_advTA_model.pkl'
# --- BARU: Path untuk model regresi dan scaler Y ---
LINREG_MODEL_SAVE_PATH = f'linreg_reg_{DEFAULT_TICKER}_advTA_model.pkl' # Asumsi pakai LinReg
SCALER_Y_REG_SAVE_PATH = f'scaler_y_classic_reg_{DEFAULT_TICKER}_advTA.pkl' 
# ---------------------------------------------------
SCALER_X_SAVE_PATH = f'scaler_X_classic_combined_{DEFAULT_TICKER}_advTA.pkl' # Scaler X tetap sama

# --- Fitur Input (DARI NOTEBOOK TRAINING - HARUS SAMA PERSIS, SEKARANG 16) ---
FEATURE_COLUMNS_INPUT = ['BBB_20_2.0', 'CMF_20', 'Close', 'Close_Diff_lag1', 
                         'Close_Diff_lag2', 'Close_Diff_lag3', 'MACD_12_26_9', 
                         'OBV', 'RSI_14', 'SMA_20', 'STOCHd_14_3_3', 'STOCHk_14_3_3', 
                         'Volume', 'Volume_lag1', 'Volume_lag2', 'Volume_lag3']
N_FEATURES_INPUT = len(FEATURE_COLUMNS_INPUT) # Akan jadi 16

st.set_page_config(page_title=f"Prediksi {DEFAULT_TICKER} (Arah & Harga)", layout="wide")
st.title(f"📊 Dashboard Prediksi Harga {DEFAULT_TICKER} (Model Klasik - Arah & Harga)")
st.markdown("Dashboard ini menggunakan model Klasik untuk memprediksi **Arah** (Naik/Turun/Sama) dan **Estimasi Harga** berikutnya.")

# --- Fungsi Helper ---

@st.cache_resource 
def load_all_models_and_scalers(ticker): # Memuat SEMUA model & scaler
    """Memuat model Klasifikasi, Regresi, dan Scaler X & Y."""
    logreg_path = f'logreg_clf_{ticker}_advTA_model.pkl'
    rf_path = f'rf_clf_{ticker}_advTA_model.pkl'
    linreg_path = f'linreg_reg_{ticker}_advTA_model.pkl' # Path model regresi
    scaler_x_path = f'scaler_X_classic_combined_{ticker}_advTA.pkl' # Sesuaikan nama jika perlu
    scaler_y_path = f'scaler_y_classic_reg_{ticker}_advTA.pkl' # Path scaler Y regresi
    
    st.write("--- Memulai Pemuatan Model & Scaler ---")
    files_ok = True
    if not os.path.exists(logreg_path): st.error(f"File model LR tidak ditemukan: {logreg_path}"); files_ok = False
    if not os.path.exists(rf_path): st.error(f"File model RF tidak ditemukan: {rf_path}"); files_ok = False
    if not os.path.exists(linreg_path): st.error(f"File model Regresi tidak ditemukan: {linreg_path}"); files_ok = False
    if not os.path.exists(scaler_x_path): st.error(f"File scaler X tidak ditemukan: {scaler_x_path}"); files_ok = False
    if not os.path.exists(scaler_y_path): st.error(f"File scaler Y (Regresi) tidak ditemukan: {scaler_y_path}"); files_ok = False
        
    if not files_ok:
        st.warning("Pastikan SEMUA file model (LogReg, RF, LinReg) dan scaler (X, Y_Reg) sudah ada dan nama ticker sesuai.")
        st.stop()

    st.write("✔️ Semua file model dan scaler ditemukan.")

    try:
        model_logreg_loaded = joblib.load(logreg_path); st.write("✔️ Model Logistic Regression (Klasifikasi) dimuat.") 
        model_rf_loaded = joblib.load(rf_path); st.write("✔️ Model Random Forest (Klasifikasi) dimuat.") 
        model_linreg_loaded = joblib.load(linreg_path); st.write("✔️ Model Linear Regression (Regresi) dimuat.") # Load model regresi
        scaler_X_loaded = joblib.load(scaler_x_path); st.write("✔️ Scaler X (Input) dimuat.") 
        scaler_y_reg_loaded = joblib.load(scaler_y_path); st.write("✔️ Scaler Y (Regresi/Diff) dimuat.") # Load scaler Y
        
        n_features_scaler = getattr(scaler_X_loaded, 'n_features_in_', None)
        # Validasi jumlah fitur yang diharapkan (16)
        if n_features_scaler is not None and n_features_scaler != N_FEATURES_INPUT: 
             st.error(f"Scaler X tidak valid (fitur: {n_features_scaler}, diharapkan: {N_FEATURES_INPUT}). Periksa FEATURE_COLUMNS_INPUT."); st.stop()
        if n_features_scaler is not None: st.write(f"✔️ Scaler X dikonfirmasi memiliki {n_features_scaler} fitur input.")
        
        st.write("--- Selesai Pemuatan ---") 
        return model_logreg_loaded, model_rf_loaded, model_linreg_loaded, scaler_X_loaded, scaler_y_reg_loaded
    except Exception as e: st.error(f"Gagal memuat model atau scaler: {e}"); st.exception(e); st.stop()

# --- Fungsi get_and_process_data (Versi v3 yang sudah diperbaiki) ---
def get_and_process_data(ticker='BTC-USD', num_days_needed=100): 
    # ... (Fungsi ini seharusnya sudah benar dari revisi sebelumnya) ...
    st.write(f"--- Memulai get_and_process_data (ticker: {ticker}, days needed: {num_days_needed}) ---") 
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=num_days_needed * 1.5)).strftime('%Y-%m-%d')
    end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d') 
    try: df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e: st.warning(f"Gagal ambil data yfinance: {e}"); return None
    if df.empty: st.warning(f"Data yfinance kosong untuk {ticker}."); return None
    st.write(f"Data mentah diambil, shape: {df.shape}") 
    df_processed = df.copy() 
    if isinstance(df_processed.columns, pd.MultiIndex):
        st.write("MultiIndex kolom. Meratakan..."); 
        try: 
            df_processed.columns = df_processed.columns.get_level_values(0).str.title()
            df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
            st.write(f"Kolom setelah perataan: {df_processed.columns.tolist()}")
        except Exception as e: st.error(f"Gagal meratakan MultiIndex: {e}."); return None
    else: 
        st.write("Menstandarkan nama kolom ke Title Case...")
        df_processed.columns = [str(col).title() for col in df_processed.columns]
        st.write(f"Kolom setelah standarisasi: {df_processed.columns.tolist()}")
    high_col_name, low_col_name, close_col_name, volume_col_name = 'High', 'Low', 'Close', 'Volume' 
    required_cols = [high_col_name, low_col_name, close_col_name, volume_col_name]
    missing_req = [col for col in required_cols if col not in df_processed.columns]
    if missing_req: st.error(f"Kolom standar hilang: {missing_req}."); return None 
    st.write(f"Kolom HLCV standar digunakan: {required_cols}")
    st.write("Menghitung fitur teknikal..."); sma_col_name, rsi_col_name, macd_col_name = 'SMA_20', 'RSI_14', 'MACD_12_26_9'
    try: df_processed[sma_col_name] = df_processed[close_col_name].rolling(window=20).mean()
    except Exception as e: st.warning(f"Error SMA: {e}")
    if PANDAS_TA_AVAILABLE:
        try: df_processed.ta.rsi(close=df_processed[close_col_name], length=14, col_names=(rsi_col_name,), append=True)
        except Exception as e: st.warning(f"Gagal RSI: {e}")
        try: df_processed.ta.macd(close=df_processed[close_col_name], col_names=(macd_col_name, f'MACDh_12_26_9', f'MACDs_12_26_9'), append=True)
        except Exception as e: st.warning(f"Gagal MACD: {e}")
        st.write("Menghitung fitur TA lanjutan...")
        adv_ta_to_calculate = {'atr': {'high': df_processed[high_col_name], 'low': df_processed[low_col_name], 'close': df_processed[close_col_name], 'length': 14, 'col_names':'ATR_14'}, 'bbands': {'close': df_processed[close_col_name], 'length': 20, 'std': 2, 'col_names': ('BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0')}, 'stoch': {'high': df_processed[high_col_name], 'low': df_processed[low_col_name], 'close': df_processed[close_col_name], 'k': 14, 'd': 3, 'smooth_k': 3, 'col_names': ('STOCHk_14_3_3', 'STOCHd_14_3_3')}, 'obv': {'close': df_processed[close_col_name], 'volume': df_processed[volume_col_name], 'col_names':'OBV'}, 'cmf': {'high': df_processed[high_col_name], 'low': df_processed[low_col_name], 'close': df_processed[close_col_name], 'volume': df_processed[volume_col_name], 'length': 20, 'col_names':'CMF_20'}}
        for func_name, params in adv_ta_to_calculate.items():
            try: getattr(df_processed.ta, func_name)(**params, append=True)
            except Exception as ta_err: st.warning(f" Gagal hitung TA ({func_name}): {ta_err}")
        st.write("Fitur TA lanjutan selesai.")
    else: st.warning("pandas_ta tidak tersedia.")
    st.write("Menghitung fitur lag...")
    target_diff_temp_col = 'Close_Diff_Temp'
    df_processed[target_diff_temp_col] = df_processed[close_col_name].diff() 
    lags_to_add = [1, 2, 3]
    for lag in lags_to_add:
        lag_col_diff = f'Close_Diff_lag{lag}'; df_processed[lag_col_diff] = df_processed[target_diff_temp_col].shift(lag)
        lag_col_vol = f'{volume_col_name}_lag{lag}'; df_processed[lag_col_vol] = df_processed[volume_col_name].shift(lag) 
    df_processed = df_processed.drop(columns=[target_diff_temp_col], errors='ignore')
    st.write(f"Fitur lag dihitung.")
    # --- PERBAIKAN: Gunakan FEATURE_COLUMNS_INPUT untuk final_cols_needed ---
    final_cols_needed = FEATURE_COLUMNS_INPUT[:] 
    if 'Close' not in final_cols_needed: final_cols_needed.append('Close') 
    available_cols = [col for col in final_cols_needed if col in df_processed.columns]
    missing_input_cols = [col for col in FEATURE_COLUMNS_INPUT if col not in available_cols] 
    if missing_input_cols:
         st.warning(f"Fitur input hilang: {missing_input_cols}.")
         # Tetapkan fitur aktual yang akan digunakan berdasarkan yang tersedia
         st.session_state.feature_columns_input_actual = [col for col in FEATURE_COLUMNS_INPUT if col in available_cols]
         if len(st.session_state.feature_columns_input_actual) != N_FEATURES_INPUT: 
              st.error(f"Jumlah fitur input ({len(st.session_state.feature_columns_input_actual)}) tidak cocok ({N_FEATURES_INPUT}).")
              return None
    else:
         # Jika semua fitur ada, gunakan list asli
         st.session_state.feature_columns_input_actual = FEATURE_COLUMNS_INPUT[:] 
    
    df_final = df_processed[available_cols].dropna() 
    st.write(f"Kolom akhir dipilih ({len(st.session_state.feature_columns_input_actual)} fitur input + Close) dan NaN dihapus.")
    if len(df_final) < 1: st.warning(f"Data tidak tersisa untuk {ticker}."); return None
    st.write(f"Shape data akhir siap pakai: {df_final.shape}") 
    st.write("--- Selesai get_and_process_data ---") 
    return df_final
# --- AKHIR FUNGSI ---

# --- Main App Logic ---
model_logreg, model_rf, model_linreg, scaler_X, scaler_y_reg = load_all_models_and_scalers(DEFAULT_TICKER) 

st.sidebar.header("Pengaturan Data")
ticker_symbol = st.sidebar.text_input("Simbol Ticker", DEFAULT_TICKER)
data_period_display = st.sidebar.selectbox(
    "Periode Grafik Historis", ["90d", "6mo", "1y", "2y"], index=2, key="periode_grafik"
)
st.sidebar.caption(f"Prediksi didasarkan pada data historis terakhir (membutuhkan ~{BUFFER_DAYS} hari data mentah).")

if st.sidebar.button("Muat Data & Prediksi", key="prediksi_button"): 
    if ticker_symbol != DEFAULT_TICKER:
         st.warning(f"Model/Scaler diload untuk {DEFAULT_TICKER}. Cek kesesuaian untuk {ticker_symbol}.")

    data_processed = get_and_process_data(ticker=ticker_symbol, num_days_needed=TIME_STEP_CONTEXT + BUFFER_DAYS) 

    if data_processed is not None and not data_processed.empty:
        st.subheader(f"Data Historis Terbaru ({ticker_symbol})")
        display_limit = int(min(90, len(data_processed))) 
        # Tampilkan fitur aktual yang digunakan
        st.dataframe(data_processed[st.session_state.feature_columns_input_actual].tail(display_limit)) 
        try: 
            last_actual_close_price = float(data_processed['Close'].iloc[-1]) 
            st.metric(label="Harga Penutupan Terakhir (Aktual)", value=f"${last_actual_close_price:,.2f}")
        except (IndexError, KeyError) as e: st.error(f"Gagal ambil harga 'Close' terakhir: {e}"); st.stop()
        except Exception as e: st.error(f"Error ambil harga terakhir: {e}"); st.stop()

        st.write("--- Memulai Proses Prediksi (Klasik - Arah & Harga) ---")
        try:
            # --- Siapkan Input (gunakan list fitur aktual dari session state) ---
            X_latest_unscaled = data_processed[st.session_state.feature_columns_input_actual].iloc[-1:] 
            st.write(f"Fitur input terakhir (unscaled): shape {X_latest_unscaled.shape}")
            if X_latest_unscaled.empty: st.error("Gagal mendapatkan baris fitur terakhir."); st.stop()
            
            # Pastikan jumlah kolom X_latest_unscaled cocok dengan scaler
            if X_latest_unscaled.shape[1] != scaler_X.n_features_in_:
                 st.error(f"Jumlah fitur data terbaru ({X_latest_unscaled.shape[1]}) tidak cocok dengan scaler ({scaler_X.n_features_in_}).")
                 st.stop()
            
            X_latest_scaled = scaler_X.transform(X_latest_unscaled) 
            st.write("Fitur input terakhir discaling.")
            
            st.write("Melakukan prediksi arah...")
            pred_logreg = model_logreg.predict(X_latest_scaled)[0] 
            proba_logreg = model_logreg.predict_proba(X_latest_scaled)[0] 
            pred_rf = model_rf.predict(X_latest_scaled)[0]
            proba_rf = model_rf.predict_proba(X_latest_scaled)[0]
            direction_map = {0: "Turun / Sama", 1: "Naik"}
            pred_direction_logreg = direction_map.get(pred_logreg, "Error")
            pred_direction_rf = direction_map.get(pred_rf, "Error")

            st.write("Melakukan prediksi harga (estimasi perubahan)...")
            scaled_diff_pred = model_linreg.predict(X_latest_scaled) 
            st.write("Prediksi Scaled Difference (Regresi):", scaled_diff_pred) 
            unscaled_diff_pred = scaler_y_reg.inverse_transform(scaled_diff_pred.reshape(-1, 1))[0, 0] 
            st.write("Predicted Difference (Unscaled):", unscaled_diff_pred)
            predicted_absolute_price = last_actual_close_price + unscaled_diff_pred 
            st.write("Predicted Price (Absolute):", predicted_absolute_price)

            st.subheader("🔮 Prediksi Harga Berikutnya")
            st.markdown("**Prediksi Arah:**")
            col1, col2 = st.columns(2)
            with col1:
                 st.metric(label="Logistic Regression", value=pred_direction_logreg)
                 st.progress(float(proba_logreg[1]))
                 st.caption(f"Prob. Naik: {proba_logreg[1]:.2%}")
                 st.caption(f"Prob. Turun/Sama: {proba_logreg[0]:.2%}")
            with col2:
                 st.metric(label="Random Forest", value=pred_direction_rf)
                 st.progress(float(proba_rf[1]))
                 st.caption(f"Prob. Naik: {proba_rf[1]:.2%}")
                 st.caption(f"Prob. Turun/Sama: {proba_rf[0]:.2%}")
            
            st.divider() 
                 
            st.markdown("**Estimasi Harga (Regresi):**")
            change = predicted_absolute_price - last_actual_close_price 
            delta_perc = (change / last_actual_close_price) * 100 if last_actual_close_price != 0 else 0
            st.metric(label="Prediksi Harga Berikutnya",
                      value=f"${predicted_absolute_price:,.2f}", 
                      delta=f"{change:,.2f} ({delta_perc:.2f}%)")
            st.caption("Estimasi ini berasal dari model regresi terpisah (misal: Linear Regression)")

            st.subheader("📊 Grafik Harga Historis & Estimasi Harga")
            days_for_plot = 90 if '90d' in data_period_display else 180 if '6mo' in data_period_display else 365 if '1y' in data_period_display else 730
            data_plot_display = get_and_process_data(ticker=ticker_symbol, num_days_needed=days_for_plot + BUFFER_DAYS) 
            
            if data_plot_display is not None and not data_plot_display.empty and 'Close' in data_plot_display.columns:
                 fig, ax = plt.subplots(figsize=(12, 6))
                 plot_data_to_show = data_plot_display.tail(days_for_plot) 
                 ax.plot(plot_data_to_show.index, plot_data_to_show['Close'].values, label=f'Harga Historis (Close) - {data_period_display}', linewidth=2)
                 last_date_processed = data_processed.index[-1] 
                 next_date = last_date_processed + pd.Timedelta(days=1) 
                 ax.plot(next_date, predicted_absolute_price, 'ro', markersize=8, label='Estimasi Berikutnya (Harga)')
                 ax.axhline(predicted_absolute_price, color='red', linestyle='--', alpha=0.7) 
                 ax.set_title(f"Grafik Harga {ticker_symbol} ({data_period_display} Terakhir) & Estimasi Harga") 
                 ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (USD)")
                 ax.legend(); ax.grid(True); fig.autofmt_xdate() 
                 st.pyplot(fig)
            else: st.warning("Tidak dapat menampilkan grafik historis.")

        except Exception as e: st.error(f"Terjadi kesalahan saat prediksi:"); st.exception(e)
    else: st.warning(f"Tidak ada data yang dapat diproses untuk {ticker_symbol}. Cek ticker atau coba periode lebih panjang.")
else: st.info("Masukkan simbol ticker dan klik 'Muat Data & Prediksi'.")

# --- END OF FILE ---