
import os, time, requests, json
import pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta
import pytz

# --- Fallback əgər tensorflow yoxdursa ---
try:
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    HAS_TF = True
except:
    HAS_TF = False
    print("⚠️ Tensorflow yoxdur, sadə rejim")

BOT_NAME = "Cavanshir83Bot"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TICKERS = ["TSLA"]

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

def get_market_countdown():
    now = datetime.now(BAKU_TZ)
    wd = now.weekday()
    if wd >=5:
        return f"⏰ Həftəsonu - Bazar bağlı", False
    today_open = now.replace(hour=17, minute=30, second=0, microsecond=0)
    if now < today_open:
        d = today_open-now
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        return f"⏰ Açılışa {h}s {m}dəq qaldı", False
    else:
        return f"🟢 Bazar AÇIQ", True

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Secrets yoxdur! Settings -> Secrets -> Actions-a 3 açarı əlavə et")
        print(text)
        return False
    try:
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data={"chat_id":TELEGRAM_CHAT_ID,"text":text}
        r=requests.post(url,data=data,timeout=10)
        print(f"Telegram: {r.status_code} {r.text[:200]}")
        return r.status_code==200
    except Exception as e:
        print(f"Telegram xətası: {e}"); return False

def get_price(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
        if df.empty: return None, 0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        price = float(df['Close'].iloc[-1])
        # sadə siqnal: MA
        df['MA20'] = df['Close'].rolling(2).mean()
        ma = float(df['MA20'].iloc[-1])
        decision = "AL" if price > ma else "SAT" if price < ma*0.98 else "GÖZLƏ"
        conf = 65.0 if decision=="AL" else 60.0
        return price, decision, conf, df
    except Exception as e:
        print(f"Data xətası: {e}")
        return None, 0, 0, None

def run():
    print("🚀 TSLA Robot GitHub 7/24")
    msg, is_open = get_market_countdown()
    print(msg)
    all_results=[]
    for ticker in TICKERS:
        price, dec, conf, df = get_price(ticker) or (None, None, None, None)
        if price is None:
            continue
        all_results.append((ticker, price, dec, conf))
        print(f"✅ {ticker} ${price:.2f} -> {dec} {conf:.0f}%")

    if not all_results:
        txt = f"🤖 {BOT_NAME} {datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')}\n{msg}\n⚠️ Data alınmadı"
        send_telegram(txt)
        return

    now_str=datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')
    lines=[f"🤖 {BOT_NAME} {now_str}", msg, "─"*20]
    for ticker, price, dec, conf in all_results:
        icon="🟢" if dec=="AL" else "🔴" if dec=="SAT" else "🟡"
        lines.append(f"📈 {ticker} ${price:.2f}")
        lines.append(f"{icon} 1 GÜN: {dec} {conf:.0f}%")
    lines.append("─"*20)
    lines.append(f"🕐 Bakı {datetime.now(BAKU_TZ).strftime('%H:%M')}")
    report="\n".join(lines)
    print(report)
    send_telegram(report)

if __name__=="__main__":
    run()
