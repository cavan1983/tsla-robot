"""
TRADE PRO V6.8 - REAL LEARNING + WORK HOURS + LIMIT SAVER NEWS
"""
import os, json, pickle, warnings, traceback, glob
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
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
    print("✅ TF + sklearn OK")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TF yoxdur: {e}")

TICKERS = ["TSLA", "KO", "AAPL", "NVDA", "MSFT"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# V6.8 - KÖHNƏ BEYİNLƏRİ SİLMİRİK - ƏSL ÖYRƏNMƏ
print("🧠 Köhnə beyinlər saxlanır - üst-üstə öyrənəcək")

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
    print(f"[1] fetch {ticker} {interval} {period}")
    for attempt in range(2):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 30:
                print(f" -> OK {len(df)} bar")
                return df
        except Exception as e:
            print(f" -> FAIL {e}")
    return None

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
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(df['LABEL'].iloc[i])

        X = np.array(X)
        y = to_categorical(y, num_classes=3)

        last_row = df.iloc[-1]
        meta = {
            'ma20': float(last_row['MA20']),
            'ma50': float(last_row['MA50']),
            'ma200': float(last_row['MA200']),
            'rsi': float(last_row['RSI']),
            'vol_change': float(last_row['VOL_CH']),
            'price': float(last_row['Close']),
            'open': float(df['Open'].iloc[-1] if 'Open' in df else last_row['Close'])
        }
        print(f" -> OK X={X.shape} MA20={meta['ma20']:.2f} RSI={meta['rsi']:.1f}")
        return X, y, scaler, meta

    except Exception as e:
        print(f" -> FAIL {e}")
        traceback.print_exc()
        return None, None, None, None

def train_for_ticker(ticker):
    horizons = {}
    if ticker == "TSLA":
        horizons = {
            "1s": {"interval": "60m", "period": "7d", "label": "1 SAAT"},
            "1g": {"interval": "1d", "period": "2y", "label": "1 GÜN"},
            "3g": {"interval": "1d", "period": "2y", "label": "3 GÜN"},
            "5g": {"interval": "1d", "period": "2y", "label": "5 GÜN"},
        }
    else:
        horizons = {"1g": {"interval": "1d", "period": "2y", "label": "1 GÜN"}}

    results = {}
    print(f"\n========== {ticker} ==========")
    for hk, cfg in horizons.items():
        try:
            print(f"\n--- {ticker} {hk} START ---")
            if hk == "1s" and not is_us_market_open():
                print(f"skip - bazar bağlı")
                continue
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

            print(f"[4] train {ticker} {hk}...")
            if TF_AVAILABLE:
                model.fit(X, y, epochs=8, batch_size=16, verbose=0)
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
    print(f"V6.8 LIMIT-SAVER - {baku} | NY {ny.strftime('%A %H:%M')} | Market: {is_us_market_open()}")

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

    # ========== NEWS FIX V6.8 - LIMIT QORUYUCU: YALNIZ YENI XEBER VARSA API ==========
    news_sent, news_head, is_new = 0, "yeni xəbər yoxdur", False
    try:
        # news_sentiment.py özü cache yoxlayır, təzədirsə API-yə vurmur
        try:
            news_sent, news_head, is_new = get_news_sentiment("TSLA")
        except TypeError:
            news_sent, news_head, is_new = get_news_sentiment()
        print(f"📰 News: {news_sent} | {news_head[:100]} | is_new={is_new}")
    except Exception as e:
        print(f"📰 News error (limit ola bilər): {e}")
        # Fallback: cache-dən oxu, API-yə vurma
        try:
            if os.path.exists("news_cache.json"):
                with open("news_cache.json","r") as f:
                    cache=json.load(f)
                    if "TSLA" in cache:
                        news_sent=cache["TSLA"].get("sentiment",0)
                        news_head=cache["TSLA"].get("headline","yeni xəbər yoxdur")
                        print(f"📰 Cache-dən bərpa: {news_head[:80]}")
        except:
            pass
    # ===============================================================================

    for r in rows:
        r["news_sentiment"] = news_sent if r.get("ticker") == "TSLA" else 0
        r["news_headline"] = news_head if r.get("ticker") == "TSLA" else "yeni xəbər yoxdur"

    if rows:
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
        with open(f"{DATA_DIR}/predictions.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📊 predictions.csv {len(rows)} sətir")

    try:
        msg = f"<b>TRADE PRO V6.8 LIMIT-SAVER</b> {baku.strftime('%d.%m %H:%M')}\n"
        msg += f"{'🟢 AÇIQ' if is_us_market_open() else '🔴 BAĞLI'} | {len(rows)} proqnoz\n"
        if is_new:
            msg += f"🆕 {news_head[:60]}\n\n"
        if "TSLA" in all_results:
            for hk in ["1s","1g","3g","5g"]:
                if hk in all_results["TSLA"]:
                    d = all_results["TSLA"][hk]
                    emoji = "🟢" if d['signal']=="AL" else "🔴" if d['signal']=="SAT" else "🟡"
                    msg += f"{emoji} TSLA {hk}: <b>{d['signal']}</b> {d['conf']:.0f}% @ ${d['price']:.2f}\n"
        send_telegram(msg)
    except Exception as e:
        print(f"Telegram: {e}")

if __name__ == "__main__":
    main()
