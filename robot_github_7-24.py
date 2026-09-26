"""
TRADE PRO V7.0 - KO ONLY + SMART LEARNING (EarlyStopping) + FIXED HORIZONS
+ 2 İLLİK BAZA + LIMIT SAVER NEWS (bütün tickerlər üçün)
"""
import os, json, pickle, warnings, traceback
from datetime import datetime, timedelta
from news_sentiment import get_news_sentiment
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
    print("✅ TF + sklearn OK")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TF yoxdur: {e}")

# YALNIZ KO
TICKERS = ["KO"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("🧠 KO üçün ağıllı öyrənmə aktiv - EarlyStopping ilə")

def get_times():
    baku = pytz.timezone("Asia/Baku")
    ny = pytz.timezone("America/New_York")
    return datetime.now(baku), datetime.now(ny)

def is_us_market_open():
    try:
        _, now_ny = get_times()
        if now_ny.weekday() >= 5: return False
        open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now_ny <= close_t
    except: return True

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        print(f"Telegram: {r.status_code}")
    except Exception as e:
        print(f"Telegram xətası: {e}")

def fetch_data(ticker, period, interval):
    db_path = f"{DATA_DIR}/db_{ticker}_{interval}_{period}.csv"
    df_db = None
    if os.path.exists(db_path):
        try:
            df_db = pd.read_csv(db_path, parse_dates=True, index_col=0)
            if df_db.empty:
                df_db = None
        except:
            df_db = None

    if df_db is not None:
        fetch_period = "5d" if interval == "1d" else "7d"
    else:
        fetch_period = period

    print(f"[1] fetch {ticker} {interval} {period} -> download {fetch_period} | db_exists={df_db is not None}")

    df_new = None
    for attempt in range(2):
        try:
            df = yf.download(ticker, period=fetch_period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 5:
                df_new = df
                break
        except Exception as e:
            print(f" -> FAIL {e}")

    if df_new is None:
        if df_db is not None:
            print(f" -> fetch alınmadı, köhnə baza qaytarılır {len(df_db)} bar")
            return df_db
        return None

    if df_db is not None:
        combined = pd.concat([df_db, df_new])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        combined.to_csv(db_path)
        print(f" -> DB yeniləndi {db_path}: {len(df_db)} -> {len(combined)} bar")
        return combined
    else:
        df_new.to_csv(db_path)
        print(f" -> İlk baza yaradıldı {db_path} {len(df_new)} bar - OK")
        return df_new

def build_model(input_shape):
    K.clear_session()
    print(f"[3] build {input_shape}")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_xy(df, horizon_key):
    print(f"[2] prepare {horizon_key} df={len(df)}")
    try:
        df = df.copy()
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
        df['Close'] = close
        df['Volume'] = vol

        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['RET'] = df['Close'].pct_change()
        df['VOL20'] = df['Volume'].rolling(20).mean()
        df['VOL_CH'] = (df['Volume'] - df['VOL20']) / df['VOL20'] * 100

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))

        shift_n = 1
        if horizon_key == "3g": shift_n = 3
        elif horizon_key == "5g": shift_n = 5
        # 1s üçün shift 1 saatdır, amma data 1h interval olduğuna görə 1 bar = 1 saat

        df['FUTURE'] = df['Close'].shift(-shift_n)
        df['CHANGE'] = (df['FUTURE'] - df['Close']) / df['Close'] * 100

        def label_change(ch):
            if ch > 1.2: return 0
            elif ch < -1.2: return 2
            else: return 1

        df['LABEL'] = df['CHANGE'].apply(label_change)
        df = df.dropna()

        if len(df) < 80:
            print(f" -> az data {len(df)}")
            return None, None, None, None

        features = ['MA20', 'MA50', 'MA200', 'RET', 'RSI', 'VOL_CH']
        df[features] = df[features].bfill().fillna(0)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features].values)

        seq_len = 60
        if len(scaled) <= seq_len:
            return None, None, None, None

        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i+seq_len])
            y.append(df['LABEL'].iloc[i+seq_len])
        
        X = np.array(X)
        y = to_categorical(y, num_classes=3)

        meta = {
            'price': float(df['Close'].iloc[-1]),
            'open': float(df['Open'].iloc[-1]),
            'ma20': float(df['MA20'].iloc[-1]),
            'ma50': float(df['MA50'].iloc[-1]),
            'ma200': float(df['MA200'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]),
            'vol_change': float(df['VOL_CH'].iloc[-1])
        }
        return X, y, scaler, meta
    except Exception as e:
        print(f"prepare error {e}")
        traceback.print_exc()
        return None, None, None, None

def train_for_ticker(ticker):
    # DÜZƏLDİLDİ - BÜTÜN HORIZONLAR
    horizons = {
        "1s": {"interval": "1h", "period": "3mo", "label": "1 SAAT"},
        "1g": {"interval": "1d", "period": "2y", "label": "1 GÜN"},
        "3g": {"interval": "1d", "period": "2y", "label": "3 GÜN"},
        "5g": {"interval": "1d", "period": "2y", "label": "5 GÜN"}
    }

    results = {}
    print(f"\n========== {ticker} ==========")
    for hk, cfg in horizons.items():
        try:
            print(f"\n--- {ticker} {hk} START ---")
            if hk == "1s" and not is_us_market_open():
                print(f"skip 1s - bazar bağlı, amma KO üçün model yenilənəcək")
                # 1s üçün bazar bağlıdırsa belə fetch etmirik, amma davam edirik
                # istəsən tam skip edə bilərsən
                # continue

            df = fetch_data(ticker, cfg["period"], cfg["interval"])
            if df is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue
            X, y, scaler, meta = prepare_xy(df, hk)
            if X is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": float(df['Close'].iloc[-1]), "open": float(df['Open'].iloc[-1]), "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue

            input_shape = (X.shape[1], X.shape[2])
            brain_path = f"{DATA_DIR}/brain_{ticker}_{hk}.keras"
            scaler_path = f"{DATA_DIR}/scaler_{ticker}_{hk}.pkl"

            model = None
            if os.path.exists(brain_path):
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model(brain_path)
                    print(f"🧠 Köhnə beyin yükləndi: {brain_path} - üstünə öyrənəcək")
                except:
                    model = build_model(input_shape)
                    print(f"🆕 Təzə model (köhnə xarab)")
            else:
                model = build_model(input_shape)
                print(f"🆕 İlk dəfə model quruldu")

            print(f"[4] train {ticker} {hk} - AĞILLI ÖYRƏNMƏ...")
            if TF_AVAILABLE:
                # AĞILLI ÖYRƏNMƏ - EarlyStopping
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
                model.fit(X, y, epochs=30, batch_size=16, verbose=0, callbacks=[early_stop])
                model.save(brain_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f" -> OK saved")

            last_seq = X[-1:]
            pred = model.predict(last_seq, verbose=0)[0]
            idx = int(np.argmax(pred))
            signals = ["AL", "GÖZLƏ", "SAT"]
            signal = signals[idx]
            conf = float(pred[idx] * 100)
            results[hk] = {
                "signal": signal, "conf": conf,
                "price": meta['price'], "open": meta['open'],
                "probs": {"AL": float(pred[0]*100), "GÖZLƏ": float(pred[1]*100), "SAT": float(pred[2]*100)},
                "meta": meta
            }
            print(f"[5] PRED {ticker} {hk}: {signal} {conf:.0f}% @ {meta['price']:.2f}")

        except Exception as e:
            print(f"[FAIL] {ticker} {hk}: {e}")
            traceback.print_exc()
            results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
    return results

def main():
    baku, ny = get_times()
    print(f"V7.0 KO ONLY + SMART - {baku} | NY {ny.strftime('%A %H:%M')} | Market: {is_us_market_open()}")

    if ny.weekday() >= 5:
        print("🔴 HƏFTƏSONU - GitHub boş işləməsin deyə çıxıram")
        return

    all_results = {}
    for ticker in TICKERS:
        try:
            res = train_for_ticker(ticker)
            all_results[ticker] = res
        except Exception as e:
            print(f"❌ {ticker} fail: {e}")
            all_results[ticker] = {}

    rows = []
    for ticker, horizons in all_results.items():
        for hk, data in horizons.items():
            meta = data.get("meta", {})
            rows.append({
                "timestamp": baku.isoformat(),
                "ticker": ticker,
                "horizon": hk,
                "signal": data.get("signal", "GÖZLƏ"),
                "confidence": data.get("conf", 50.0),
                "price": data.get("price", 0.0),
                "open": data.get("open", 0.0),
                "AL": data.get("probs", {}).get("AL", 33.0),
                "GÖZLƏ": data.get("probs", {}).get("GÖZLƏ", 50.0),
                "SAT": data.get("probs", {}).get("SAT", 17.0),
                "MA20": meta.get("ma20", 0),
                "MA50": meta.get("ma50", 0),
                "MA200": meta.get("ma200", 0),
                "RSI": meta.get("rsi", 0),
                "VolumeChange": meta.get("vol_change", 0),
                "VolumeStatus": "Yüksək" if meta.get("vol_change",0) > 10 else "Orta" if meta.get("vol_change",0) > -10 else "Zəif"
            })

    # DÜZƏLDİLDİ - NEWS BÜTÜN TICKERLƏR ÜÇÜN (İNDİ TƏK KO)
    news_data = {}
    for ticker in TICKERS:
        try:
            sent, head, is_new = get_news_sentiment(ticker)
            news_data[ticker] = (sent, head, is_new)
            print(f"📰 News {ticker}: {sent} | {head[:100]} | is_new={is_new}")
        except Exception as e:
            print(f"📰 News error {ticker}: {e}")
            news_data[ticker] = (0, "yeni xəbər yoxdur", False)

    for r in rows:
        t = r.get("ticker")
        if t in news_data:
            r["news_sentiment"] = news_data[t][0]
            r["news_headline"] = news_data[t][1]
            r["is_new"] = news_data[t][2]
        else:
            r["news_sentiment"] = 0
            r["news_headline"] = "yeni xəbər yoxdur"

    if rows:
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
        with open(f"{DATA_DIR}/predictions.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📊 predictions.csv {len(rows)} sətir")

    try:
        msg = f"<b>TRADE PRO V7.0 KO SMART</b> {baku.strftime('%d.%m %H:%M')}\n"
        msg += f"{'🟢 AÇIQ' if is_us_market_open() else '🔴 BAĞLI'} | {len(rows)} proqnoz\n"
        for ticker in TICKERS:
            if ticker in news_data and news_data[ticker][2]:
                msg += f"🆕 {ticker}: {news_data[ticker][1][:60]}\n"
        msg += "\n"
        if "KO" in all_results:
            for hk in ["1s","1g","3g","5g"]:
                if hk in all_results["KO"]:
                    d = all_results["KO"][hk]
                    emoji = "🟢" if d['signal']=="AL" else "🔴" if d['signal']=="SAT" else "🟡"
                    msg += f"{emoji} KO {hk}: <b>{d['signal']}</b> {d['conf']:.0f}% @ ${d['price']:.2f}\n"
        send_telegram(msg)
    except Exception as e:
        print(f"Telegram: {e}")

if __name__ == "__main__":
    main()
