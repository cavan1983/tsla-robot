"""
TRADE PRO V6.2 OPTIMIZED - TSLA 4 proqnoz, digərləri 1g
GitHub limitini keçmir, canlı qiymət hamısı üçün REAL
"""
import os, json, pickle, warnings, traceback
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
    print("✅ TF + sklearn OK")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TF yoxdur: {e}")

# OPTIMIZED CONFIG
TICKERS = ["TSLA", "KO", "AAPL", "NVDA", "MSFT"]
BASE_TICKER = "TSLA"

# TSLA üçün 4 horizon, digərləri üçün yalnız 1g (vaxta qənaət)
def get_horizons_for_ticker(ticker):
    if ticker == "TSLA":
        return {
            "1s": {"interval": "60m", "period": "7d", "label": "1 SAAT"},
            "1g": {"interval": "1d", "period": "1y", "label": "1 GÜN"},
            "3g": {"interval": "1d", "period": "2y", "label": "3 GÜN"},
            "5g": {"interval": "1d", "period": "2y", "label": "5 GÜN"},
        }
    else:
        # KO, AAPL, NVDA, MSFT üçün yalnız 1g - transfer ilə tez öyrənir
        return {
            "1g": {"interval": "1d", "period": "1y", "label": "1 GÜN"},
        }

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_times():
    baku = pytz.timezone("Asia/Baku")
    ny = pytz.timezone("America/New_York")
    return datetime.now(baku), datetime.now(ny)

def is_us_market_open():
    try:
        _, now_ny = get_times()
        if now_ny.weekday() >= 5:
            return False
        open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now_ny <= close_t
    except:
        return True

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        print(f"Telegram: {r.status_code}")
    except Exception as e:
        print(f"Telegram xətası: {e}")

def fetch_data(ticker, period, interval):
    for attempt in range(2):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 30:
                print(f"✅ {ticker} {interval} {len(df)} bar")
                return df
        except Exception as e:
            print(f"⚠️ {ticker} {attempt}: {e}")
    return None

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_xy(df, horizon_key):
    try:
        df = df.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['RET'] = df['Close'].pct_change()
        df = df.dropna()
        if len(df) < 80:
            return None, None, None
        features = ['Close', 'Volume', 'MA20', 'MA50', 'RET']
        for f in features:
            if f not in df.columns:
                return None, None, None
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])
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
        seq_len = 20
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            label = int(df['LABEL'].iloc[i])
            one_hot = [0,0,0]
            one_hot[label] = 1
            y.append(one_hot)
        if len(X) < 10:
            return None, None, None
        return np.array(X), np.array(y), scaler
    except Exception as e:
        print(f"prepare_xy {horizon_key}: {e}")
        return None, None, None

def train_for_ticker(ticker):
    results = {}
    horizons = get_horizons_for_ticker(ticker)
    print(f"\n{'='*10} {ticker} {list(horizons.keys())} {'='*10}")
    for hk, cfg in horizons.items():
        try:
            if hk == "1s" and not is_us_market_open():
                print(f"{ticker} {hk} skip - bağlı")
                continue
            df = fetch_data(ticker, cfg["period"], cfg["interval"])
            if df is None:
                print(f"❌ {ticker} {hk} data yox")
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue
            X, y, scaler = prepare_xy(df, hk)
            if X is None:
                print(f"❌ {ticker} {hk} XY yox")
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": float(df['Close'].iloc[-1]), "open": float(df['Open'].iloc[-1]), "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue
            input_shape = (X.shape[1], X.shape[2])
            brain_path = f"{DATA_DIR}/brain_{ticker}_{hk}.keras"
            scaler_path = f"{DATA_DIR}/scaler_{ticker}_{hk}.pkl"
            base_brain_path = f"{DATA_DIR}/brain_{BASE_TICKER}_{hk}.keras"
            model = None
            if ticker != BASE_TICKER and os.path.exists(base_brain_path):
                try:
                    base = load_model(base_brain_path)
                    new_model = build_model(input_shape)
                    try:
                        if base.get_weights()[0].shape == new_model.get_weights()[0].shape:
                            new_model.set_weights(base.get_weights())
                            print(f"🔄 {ticker} {hk} transfer {BASE_TICKER}-dan")
                    except:
                        pass
                    new_model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
                    model = new_model
                except Exception as e:
                    print(f"Transfer alınmadı: {e}")
            if model is None:
                if os.path.exists(brain_path):
                    try:
                        model = load_model(brain_path)
                        print(f"📂 {ticker} {hk} mövcud")
                    except:
                        model = build_model(input_shape)
                else:
                    model = build_model(input_shape)
            try:
                if TF_AVAILABLE:
                    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
                    model.save(brain_path)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                    print(f"✅ {ticker} {hk} saved")
            except Exception as e:
                print(f"⚠️ Train {ticker} {hk}: {e}")
                try:
                    model.save(brain_path)
                except:
                    pass
            try:
                last_seq = X[-1:]
                pred = model.predict(last_seq, verbose=0)[0]
                idx = int(np.argmax(pred))
                signals = ["AL", "GÖZLƏ", "SAT"]
                signal = signals[idx]
                conf = float(pred[idx] * 100)
                current_price = float(df['Close'].iloc[-1])
                open_price = float(df['Open'].iloc[-1])
                results[hk] = {
                    "signal": signal, "conf": conf,
                    "price": current_price, "open": open_price,
                    "probs": {"AL": float(pred[0]*100), "GÖZLƏ": float(pred[1]*100), "SAT": float(pred[2]*100)}
                }
                print(f"🔮 {ticker} {hk}: {signal} {conf:.0f}% @ {current_price:.2f}")
            except Exception as e:
                print(f"Pred {ticker} {hk}: {e}")
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": float(df['Close'].iloc[-1]), "open": float(df['Open'].iloc[-1]), "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
        except Exception as e:
            print(f"❌ {ticker} {hk}: {e}")
            traceback.print_exc()
            results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
    return results

def main():
    baku, ny = get_times()
    print(f"V6.2 OPTIMIZED - {baku} | TSLA 4 proqnoz, digərləri 1g")
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
                "SAT": data.get("probs", {}).get("SAT", 17.0)
            })
    if rows:
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
        with open(f"{DATA_DIR}/predictions.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📊 predictions.csv {len(rows)} sətir")
        print(df_pred.to_string())
    # Telegram
    try:
        msg = f"<b>TRADE PRO V6.2</b> {baku.strftime('%d.%m %H:%M')}\n"
        msg += f"{'🟢 AÇIQ' if is_us_market_open() else '🔴 BAĞLI'} | {len(rows)} proqnoz\n\n"
        if "TSLA" in all_results:
            for hk in ["1s","1g","3g","5g"]:
                if hk in all_results["TSLA"]:
                    d = all_results["TSLA"][hk]
                    msg += f"TSLA {hk}: <b>{d['signal']}</b> {d['conf']:.0f}% @ ${d['price']:.2f}\n"
        msg += f"\n"
        for t in ["KO","AAPL","NVDA","MSFT"]:
            if t in all_results and "1g" in all_results[t]:
                d = all_results[t]["1g"]
                msg += f"{t}: {d['signal']} {d['conf']:.0f}% @ ${d['price']:.2f}\n"
        msg += f"\n⏰ Bakı {baku.strftime('%H:%M:%S')} | Real qiymət app-də"
        send_telegram(msg)
    except Exception as e:
        print(f"Telegram: {e}")

if __name__ == "__main__":
    main()
