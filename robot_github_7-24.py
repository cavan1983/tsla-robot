"""
TRADE PRO V6 - MULTI-TICKER + TRANSFER LEARNING
TSLA təcrübəsini digər tikerlərə transfer edir
"""
import os, json, pickle, warnings
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
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

TICKERS = ["TSLA", "KO", "AAPL", "NVDA", "MSFT"]
BASE_TICKER = "TSLA"
HORIZONS = {
    "1s": {"interval": "60m", "period": "7d", "pred_hours": 1},
    "1g": {"interval": "1d", "period": "1y", "pred_days": 1},
    "3g": {"interval": "1d", "period": "2y", "pred_days": 3},
    "5g": {"interval": "1d", "period": "2y", "pred_days": 5},
}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_times():
    baku = pytz.timezone("Asia/Baku")
    ny = pytz.timezone("America/New_York")
    return datetime.now(baku), datetime.now(ny)

def is_us_market_open():
    _, now_ny = get_times()
    if now_ny.weekday() >= 5:
        return False
    open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_ny <= close_t

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"Telegram: {r.status_code}")
    except Exception as e:
        print(f"Telegram xətası: {e}")

def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"{ticker} {interval} alınmadı: {e}")
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

def build_model_transfer(base_model_path, input_shape):
    try:
        if os.path.exists(base_model_path):
            base = load_model(base_model_path)
            new_model = build_model(input_shape)
            try:
                new_model.set_weights(base.get_weights())
                print(f"🔄 Transfer: {base_model_path} -> yeni")
            except:
                print(f"⚠️ Transfer shape uyğun gəlmədi")
            new_model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            return new_model
    except Exception as e:
        print(f"Transfer xətası: {e}")
    return build_model(input_shape)

def prepare_xy(df, horizon_key):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['RET'] = df['Close'].pct_change()
    df = df.dropna()
    if len(df) < 100:
        return None, None, None
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RET']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    pred_days = HORIZONS[horizon_key].get("pred_days", 0)
    shift_n = pred_days if pred_days else 1
    future_close = df['Close'].shift(-shift_n)
    df['FUTURE'] = future_close
    df['CHANGE'] = (df['FUTURE'] - df['Close']) / df['Close'] * 100
    def label_change(ch):
        if ch > 1.5: return 0
        elif ch < -1.5: return 2
        else: return 1
    df['LABEL'] = df['CHANGE'].apply(label_change)
    df = df.dropna()
    seq_len = 30
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        label = df['LABEL'].iloc[i]
        one_hot = [0,0,0]
        one_hot[label] = 1
        y.append(one_hot)
    if len(X) < 10:
        return None, None, None
    return np.array(X), np.array(y), scaler

def train_for_ticker(ticker):
    results = {}
    print(f"\n==== {ticker} ====")
    for hk, cfg in HORIZONS.items():
        if hk == "1s" and not is_us_market_open():
            print(f"{ticker} {hk} skip - bazar bağlı")
            continue
        df = fetch_data(ticker, cfg["period"], cfg["interval"])
        if df is None or len(df) < 60:
            print(f"⚠️ {ticker} {hk} alınmadı")
            continue
        X, y, scaler = prepare_xy(df, hk)
        if X is None:
            print(f"⚠️ {ticker} {hk} data az")
            continue
        input_shape = (X.shape[1], X.shape[2])
        brain_path = f"{DATA_DIR}/brain_{ticker}_{hk}.keras"
        scaler_path = f"{DATA_DIR}/scaler_{ticker}_{hk}.pkl"
        base_brain_path = f"{DATA_DIR}/brain_{BASE_TICKER}_{hk}.keras"
        if ticker != BASE_TICKER and os.path.exists(base_brain_path) and not os.path.exists(brain_path):
            print(f"🧠 {ticker} {hk} üçün {BASE_TICKER} transfer...")
            model = build_model_transfer(base_brain_path, input_shape)
        elif os.path.exists(brain_path) and TF_AVAILABLE:
            try:
                model = load_model(brain_path)
                print(f"📂 {ticker} {hk} yükləndi")
            except:
                model = build_model(input_shape)
        else:
            model = build_model(input_shape)
        if TF_AVAILABLE:
            try:
                model.fit(X, y, epochs=8, batch_size=16, verbose=0)
                model.save(brain_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✅ {ticker} {hk} -> {brain_path}")
            except Exception as e:
                print(f"❌ {ticker} {hk} train: {e}")
                continue
        try:
            last_seq = X[-1:]
            pred = model.predict(last_seq, verbose=0)[0]
            idx = np.argmax(pred)
            signals = ["AL", "GÖZLƏ", "SAT"]
            signal = signals[idx]
            conf = float(pred[idx] * 100)
            current_price = float(df['Close'].iloc[-1])
            open_price = float(df['Open'].iloc[-1])
            results[hk] = {
                "signal": signal,
                "conf": conf,
                "price": current_price,
                "open": open_price,
                "probs": {"AL": float(pred[0]*100), "GÖZLƏ": float(pred[1]*100), "SAT": float(pred[2]*100)}
            }
            print(f"✅ {ticker} {hk}: {signal} {conf:.0f}% @ {current_price}")
        except Exception as e:
            print(f"Proqnoz {ticker} {hk}: {e}")
    return results

def main():
    baku, ny = get_times()
    print(f"Cavansir83Bot - {TICKERS} - SMART V6 MULTI")
    print(f"Bakı {baku.strftime('%H:%M')} | NY {ny.strftime('%H:%M')}")
    all_results = {}
    for ticker in TICKERS:
        res = train_for_ticker(ticker)
        all_results[ticker] = res
    rows = []
    for ticker, horizons in all_results.items():
        for hk, data in horizons.items():
            rows.append({
                "timestamp": baku.isoformat(),
                "ticker": ticker,
                "horizon": hk,
                "signal": data["signal"],
                "confidence": data["conf"],
                "price": data["price"],
                "open": data["open"],
                "AL": data["probs"]["AL"],
                "GÖZLƏ": data["probs"]["GÖZLƏ"],
                "SAT": data["probs"]["SAT"]
            })
    if rows:
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
        with open(f"{DATA_DIR}/predictions.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📊 predictions.csv: {len(rows)} sətir")
    if "TSLA" in all_results and all_results["TSLA"]:
        msg = f"<b>Cavanshir83Bot V6 MULTI</b> {baku.strftime('%d.%m %H:%M')}\n"
        msg += f"{'🟢 AÇIQ' if is_us_market_open() else '🔴 BAĞLI'}\n\n"
        for hk in ["1s","1g","3g","5g"]:
            if hk in all_results["TSLA"]:
                d = all_results["TSLA"][hk]
                msg += f"TSLA {hk}: <b>{d['signal']}</b> {d['conf']:.0f}%\n"
        msg += f"\n"
        for t in ["KO","AAPL","NVDA","MSFT"]:
            if t in all_results and "1g" in all_results[t]:
                d = all_results[t]["1g"]
                msg += f"{t}: {d['signal']} {d['conf']:.0f}% @ ${d['price']:.2f}\n"
        msg += f"\n⏰ Bakı {baku.strftime('%H:%M:%S')}"
        send_telegram(msg)
    else:
        send_telegram(f"<b>V6</b> {baku.strftime('%d.%m %H:%M')} Data alınmadı")

if __name__ == "__main__":
    main()
