
import os, time, requests, json
import pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Tokenlər GitHub Secrets-dən gəlir
BOT_NAME = "Cavanshir83Bot"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8909079473:AAHX7XMiGwou8sbd5Rs1B-0CU0LBUgHc32w")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1434358288")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "dajgcb1r01qhhp593b1gdajgcb1r01qhhp593b20")
TICKERS = ["TSLA"]

BAKU_TZ = pytz.timezone("Asia/Baku")
DRIVE_PATH = "./data"
DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

FINNHUB_CACHE_FILE = f"{DRIVE_PATH}/finnhub_cache.json"
PREDICTION_LOG = f"{DRIVE_PATH}/predictions_self_learning.csv"

HORIZONS = {
    "1s": {"days": 0.04, "label": "1 SAAT", "interval": "60m", "period": "60d", "shift": 1},
    "1g": {"days": 1, "label": "1 GÜN", "interval": "1d", "period": "2y", "shift": 1},
    "3g": {"days": 3, "label": "3 GÜN", "interval": "1d", "period": "2y", "shift": 3},
    "5g": {"days": 5, "label": "5 GÜN", "interval": "1d", "period": "2y", "shift": 5},
}

def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def load_cache():
    try:
        if os.path.exists(FINNHUB_CACHE_FILE):
            with open(FINNHUB_CACHE_FILE, "r") as jf:
                return json.load(jf)
    except: pass
    return {}

def save_cache(cache):
    try:
        with open(FINNHUB_CACHE_FILE, "w") as jf:
            json.dump(cache, jf)
    except: pass

def is_cache_valid(ts):
    try:
        ct = datetime.fromisoformat(ts)
        now = datetime.now(BAKU_TZ)
        return (now-ct).total_seconds()/3600 < 6
    except: return False

def get_market_countdown():
    now = datetime.now(BAKU_TZ)
    wd = now.weekday()
    if wd >= 5:
        days_ahead = 7-wd
        next_open = (now+timedelta(days=days_ahead)).replace(hour=17, minute=30, second=0, microsecond=0)
        d = next_open-now
        h = int(d.total_seconds()//3600)
        m = int((d.total_seconds()%3600)//60)
        msg = f"⏰ Açılışa {h}s {m}dəq (həftəsonu)"
        return msg, False, d
    today_open = now.replace(hour=17, minute=30, second=0, microsecond=0)
    today_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if now < today_open:
        d = today_open-now
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        msg = f"⏰ Açılışa {h}s {m}dəq qaldı"
        return msg, False, d
    else:
        d = today_close-now
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        msg = f"🟢 Bazar AÇIQ | Bağlanmaya {h}s {m}dəq"
        return msg, True, d

def is_us_market_open():
    _, is_open, _ = get_market_countdown()
    return is_open

def get_finnhub_data(ticker):
    cache = load_cache()
    if ticker in cache and is_cache_valid(cache[ticker].get('timestamp','')):
        return cache[ticker].get('sentiment_score',0.0), cache[ticker].get('news_summary','Neytral'), cache[ticker].get('analyst_score',0.0), cache[ticker].get('target_info','')
    sentiment_score=0.0; news_summary="Neytral"; analyst_score=0.0; target_info=""
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if r and 'companyNewsScore' in r:
            sentiment_score = float(r.get('companyNewsScore',0))
        time.sleep(1)
    except: pass
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}"
        rec = requests.get(url, timeout=10).json()
        if rec and isinstance(rec,list) and len(rec)>0:
            latest=rec[0]
            buy=latest.get('buy',0)+latest.get('strongBuy',0)*1.5
            sell=latest.get('sell',0)+latest.get('strongSell',0)*1.5
            hold=latest.get('hold',0)
            total=buy+sell+hold
            if total>0:
                analyst_score=(buy-sell)/total
        time.sleep(1)
    except: pass
    cache[ticker]={'sentiment_score':sentiment_score,'news_summary':news_summary,'analyst_score':analyst_score,'target_info':target_info,'timestamp':datetime.now(BAKU_TZ).isoformat()}
    save_cache(cache)
    return sentiment_score, news_summary, analyst_score, target_info

def get_data(ticker, horizon_key):
    cfg=HORIZONS[horizon_key]
    try:
        df=yf.download(ticker, period=cfg["period"], interval=cfg["interval"], auto_adjust=True, progress=False)
        df=clean_df(df)
        if df.empty or len(df)<30: return None
        return df
    except: return None

def add_features(df):
    df['MA20']=df['Close'].rolling(20).mean()
    df['MA50']=df['Close'].rolling(50).mean()
    df['MA200']=df['Close'].rolling(200).mean() if len(df)>200 else df['Close'].rolling(50).mean()
    delta=df['Close'].diff()
    gain=delta.where(delta>0,0).rolling(14).mean()
    loss=-delta.where(delta<0,0).rolling(14).mean()
    df['RSI']=100-(100/(1+gain/loss))
    exp12=df['Close'].ewm(span=12).mean()
    exp26=df['Close'].ewm(span=26).mean()
    df['MACD']=exp12-exp26
    df['MACD_Signal']=df['MACD'].ewm(span=9).mean()
    bb_mid=df['Close'].rolling(20).mean()
    bb_std=df['Close'].rolling(20).std()
    df['BB_Upper']=bb_mid+2*bb_std
    df['BB_Lower']=bb_mid-2*bb_std
    df['BB_Pos']=(df['Close']-df['BB_Lower'])/(df['BB_Upper']-df['BB_Lower'])
    df['Volatility']=df['Close'].pct_change().rolling(20).std()
    df['Volume_MA']=df['Volume'].rolling(20).mean()
    df['Volume_Ratio']=df['Volume']/df['Volume_MA']
    try:
        spy=clean_df(yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False))
        vix=clean_df(yf.download("^VIX", period="2y", interval="1d", auto_adjust=True, progress=False))
        df['SPY']=spy['Close'].reindex(df.index).ffill().bfill()
        df['VIX']=vix['Close'].reindex(df.index).ffill().bfill()
    except:
        df['SPY']=df['Close']; df['VIX']=15
    df['SPY_Trend']=df['SPY'].pct_change(20, fill_method=None)
    return df

def prepare_dataset(df, ticker, horizon_key):
    cfg=HORIZONS[horizon_key]
    shift=cfg["shift"]
    df=add_features(df.copy())
    news_score, news_sum, analyst_score, analyst_sum = get_finnhub_data(ticker)
    df['News_Score']=news_score
    df['Analyst_Score']=analyst_score
    df['Fund_Score']=0
    df['Future']=df['Close'].shift(-shift)/df['Close']-1
    if horizon_key=="1s": thr=0.005
    elif horizon_key=="1g": thr=0.015
    elif horizon_key=="3g": thr=0.025
    else: thr=0.03
    def etiket(x):
        if x>thr: return 2
        elif x<-thr: return 0
        else: return 1
    df['Label']=df['Future'].apply(etiket)
    df.dropna(inplace=True)
    features=['Close','MA20','MA50','RSI','MACD','BB_Pos','Volatility','Volume_Ratio','SPY','VIX','News_Score','Analyst_Score','Fund_Score','SPY_Trend']
    features=[f for f in features if f in df.columns]
    df_num=df[features].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df_num)<60: return None,None,None
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(df_num)
    min_len=min(len(scaled), len(df))
    scaled=scaled[-min_len:]
    labels=df['Label'].iloc[-min_len:].values
    lookback=30 if horizon_key=="1s" else 60
    X,y=[],[]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i]); y.append(labels[i])
    X,y=np.array(X),np.array(y)
    if len(X)<30: return None,None,None
    return X,y,(df,features,scaler)

def load_eval_log():
    if os.path.exists(PREDICTION_LOG):
        try: return pd.read_csv(PREDICTION_LOG)
        except: return pd.DataFrame()
    return pd.DataFrame()

def train_model(ticker, horizon_key, X, y, df_info):
    df, features, scaler = df_info
    brain_file=f"{DRIVE_PATH}/brain_{ticker}_{horizon_key}_self.h5"
    model=None
    if os.path.exists(brain_file):
        try:
            model=load_model(brain_file)
            if model.input_shape[-1]!=len(features):
                os.remove(brain_file); model=None
            else:
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        except:
            if os.path.exists(brain_file): os.remove(brain_file); model=None
    if model is None:
        model=Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(64, return_sequences=True), Dropout(0.25), LSTM(32), Dropout(0.25), Dense(16, activation='relu'), Dense(3, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    sample_weight=np.ones(len(y))
    model.fit(X,y,epochs=10,batch_size=32,verbose=0,sample_weight=sample_weight)
    model.save(brain_file)
    return model,scaler,features,df

def predict_horizon(ticker, horizon_key):
    df_raw=get_data(ticker,horizon_key)
    if df_raw is None: return None
    prepared=prepare_dataset(df_raw,ticker,horizon_key)
    if prepared[0] is None: return None
    X,y,df_info=prepared
    df,features,scaler=df_info
    brain_file=f"{DRIVE_PATH}/brain_{ticker}_{horizon_key}_self.h5"
    if not os.path.exists(brain_file):
        train_model(ticker,horizon_key,X,y,df_info)
    model=load_model(brain_file)
    lookback=30 if horizon_key=="1s" else 60
    last_n=scaler.transform(df[features].apply(pd.to_numeric, errors='coerce').dropna().tail(lookback))
    X_pred=np.array([last_n])
    probs=model.predict(X_pred,verbose=0)[0]
    pred=int(np.argmax(probs)); conf=float(np.max(probs)*100)
    mapping={0:"SAT",1:"GÖZLƏ",2:"AL"}
    last=df.iloc[-1].to_dict()
    return mapping[pred],conf,probs,last

def log_prediction(ticker,horizon_key,decision,conf,price,probs):
    now=datetime.now(BAKU_TZ)
    cfg=HORIZONS[horizon_key]
    target_time=now+timedelta(hours=1) if horizon_key=="1s" else now+timedelta(days=cfg["days"])
    row={"Date":now.strftime("%Y-%m-%d %H:%M"), "TargetDate":target_time.strftime("%Y-%m-%d %H:%M"), "Ticker":ticker, "Horizon":horizon_key, "Decision":decision, "Confidence":f"{conf:.0f}%", "PriceAtPred":price, "Checked":False}
    df=pd.DataFrame([row])
    if os.path.exists(PREDICTION_LOG):
        df.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG, index=False)

def get_overall_stats():
    try:
        if not os.path.exists(PREDICTION_LOG): return "📊 Hələ yoxdur",0,0,0
        df=pd.read_csv(PREDICTION_LOG)
        checked=df[df['Checked']==True] if 'Checked' in df.columns else pd.DataFrame()
        if checked.empty: return f"📊 {len(df)} proqnoz gözləyir",0,len(df),0
        total=len(checked); correct=len(checked[checked['Correct']==True]) if 'Correct' in checked.columns else 0
        acc=correct/total*100 if total>0 else 0
        return f"🏆 Uğur: {acc:.0f}% ({correct}/{total})",acc,total,correct
    except: return "📊 Xəta",0,0,0

def send_telegram(text):
    try:
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data={"chat_id":TELEGRAM_CHAT_ID,"text":text}
        r=requests.post(url,data=data,timeout=10)
        print(f"Telegram: {r.status_code}")
        return r.status_code==200
    except Exception as e:
        print(f"Telegram xətası: {e}"); return False

def format_report(all_results):
    market_msg,_,_=get_market_countdown()
    overall_msg,_,_,_=get_overall_stats()
    now_str=datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')
    lines=[f"🤖 {BOT_NAME} {now_str}", market_msg, overall_msg, "─"*20]
    for ticker in sorted(set([r[0] for r in all_results])):
        ticker_results=[r for r in all_results if r[0]==ticker]
        last_price=ticker_results[0][3].get('Close',0) if ticker_results else 0
        lines.append(f"📈 {ticker} ${last_price:.2f}")
        for _,hk,dec,conf,probs,last in ticker_results:
            cfg=HORIZONS[hk]
            icon="🟢" if dec=="AL" else "🔴" if dec=="SAT" else "🟡"
            lines.append(f"{icon} {cfg['label']}: {dec} {conf:.0f}%")
    lines.append("─"*20)
    lines.append(f"🕐 Bakı {datetime.now(BAKU_TZ).strftime('%H:%M')}")
    return "\n".join(lines)

def run():
    print(f"🚀 {BOT_NAME} - {TICKERS}")
    msg,_,_=get_market_countdown()
    print(msg)
    all_results=[]
    for ticker in TICKERS:
        for hk in ["1s","1g","3g","5g"]:
            if hk=="1s" and "USD" not in ticker and not is_us_market_open():
                continue
            try:
                res=predict_horizon(ticker,hk)
                if res is None: continue
                dec,conf,probs,last=res
                log_prediction(ticker,hk,dec,conf,last.get('Close',0),probs)
                all_results.append((ticker,hk,dec,conf,probs,last))
                print(f"✅ {ticker} {hk}: {dec} {conf:.0f}%")
            except Exception as e:
                print(f"❌ {ticker} {hk}: {e}")
    if all_results:
        report=format_report(all_results)
        print("\n"+report)
        send_telegram(report)

if __name__=="__main__":
    run()
