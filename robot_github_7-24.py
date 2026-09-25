import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING'] = '0'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import time, requests, json, glob, pickle
import pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    HAS_TF = True
except Exception as e:
    print(f"TF yoxdur: {e}")
    HAS_TF = False

BOT_NAME = "Cavanshir83Bot"

# --- TƏHLÜKƏSİZLİK: Default token YOXDUR ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
# 1. TICKERS əlavə et
TICKERS = ["TSLA", "KO", "AAPL", "NVDA", "MSFT"]
BASE_TICKER = "TSLA"

def build_model_transfer(base_model_path, input_shape):
    if os.path.exists(base_model_path):
        base = load_model(base_model_path)
        new_model = build_model(input_shape)
        try:
            new_model.set_weights(base.get_weights())
            print(f"🔄 Transfer: {base_model_path}")
        except:
            pass
        new_model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return new_model
    return build_model(input_shape)

def train_for_ticker(ticker):
    for hk, cfg in HORIZONS.items():
        brain_path = f"data/brain_{ticker}_{hk}.keras"
        base_brain_path = f"data/brain_{BASE_TICKER}_{hk}.keras"
        
        if ticker != BASE_TICKER and not os.path.exists(brain_path) and os.path.exists(base_brain_path):
            print(f"🧠 {ticker} {hk} üçün {BASE_TICKER} transfer...")
            model = build_model_transfer(base_brain_path, input_shape)
        # ... rest train
BAKU_TZ = pytz.timezone("Asia/Baku")
NY_TZ = pytz.timezone("America/New_York")
DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

FINNHUB_CACHE_FILE = f"{DATA_PATH}/finnhub_cache.json"
PREDICTION_LOG = f"{DATA_PATH}/predictions.csv"

HORIZONS = {
    "1s": {"days": 0.04, "label": "1 SAAT", "interval": "60m", "period": "7d", "shift": 1},
    "1g": {"days": 1, "label": "1 GÜN", "interval": "1d", "period": "2y", "shift": 1},
    "3g": {"days": 3, "label": "3 GÜN", "interval": "1d", "period": "2y", "shift": 3},
    "5g": {"days": 5, "label": "5 GÜN", "interval": "1d", "period": "2y", "shift": 5},
}

SPY_CACHE = None

def clean_df(df):
    if df is None or df.empty:
        return df
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

def is_us_market_open():
    now_ny = datetime.now(NY_TZ)
    if now_ny.weekday() >= 5:
        return False
    open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_ny <= close_t

def get_market_countdown():
    now_ny = datetime.now(NY_TZ)
    if now_ny.weekday() >= 5:
        days_ahead = 7 - now_ny.weekday()
        next_open = (now_ny + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
        d = next_open - now_ny
        h = int(d.total_seconds()//3600)
        return f"⏰ Açılışa {h}s (həftəsonu)", False, d
    
    open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now_ny < open_t:
        d = open_t - now_ny
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        return f"⏰ Açılışa {h}s {m}dəq qaldı", False, d
    elif now_ny <= close_t:
        d = close_t - now_ny
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        return f"🟢 Bazar AÇIQ | Bağlanmaya {h}s {m}dəq", True, d
    else:
        next_day = now_ny + timedelta(days=1)
        if next_day.weekday() >= 5:
            next_day += timedelta(days=(7-next_day.weekday()))
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        d = next_day - now_ny
        h = int(d.total_seconds()//3600)
        return f"🔴 Bağlıdır | Açılışa {h}s", False, d

def get_finnhub_data(ticker):
    cache = load_cache()
    if ticker in cache and is_cache_valid(cache[ticker].get('timestamp','')):
        return cache[ticker].get('sentiment_score',0.0), cache[ticker].get('news_summary','Neytral'), cache[ticker].get('analyst_score',0.0)
    sentiment_score=0.0; news_summary="Neytral"; analyst_score=0.0
    if not FINNHUB_API_KEY:
        return sentiment_score, news_summary, analyst_score
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if r and 'companyNewsScore' in r:
            sentiment_score = float(r.get('companyNewsScore',0))
            if sentiment_score>0.2: news_summary=f"Pozitiv {sentiment_score:+.2f}"
            elif sentiment_score<-0.2: news_summary=f"Negativ {sentiment_score:.2f}"
            else: news_summary=f"Neytral {sentiment_score:+.2f}"
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
    except: pass
    cache[ticker]={'sentiment_score':sentiment_score,'news_summary':news_summary,'analyst_score':analyst_score,'timestamp':datetime.now(BAKU_TZ).isoformat()}
    save_cache(cache)
    return sentiment_score, news_summary, analyst_score

def get_current_price(ticker):
    real_price = None
    # 1. Finnhub
    try:
        if FINNHUB_API_KEY:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=10).json()
            if r and 'c' in r and r['c'] > 0:
                return float(r['c'])
    except: pass
    # 2. yfinance
    try:
        t = yf.Ticker(ticker)
        if hasattr(t, 'fast_info') and t.fast_info and 'last_price' in t.fast_info:
            v = t.fast_info['last_price']
            if v: return float(v)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        hist = t.history(period="1d", interval="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except: pass
    return None

def get_data(ticker, horizon_key):
    cfg=HORIZONS[horizon_key]
    for attempt in range(3):
        try:
            df=yf.download(ticker, period=cfg["period"], interval=cfg["interval"], auto_adjust=True, progress=False, timeout=30)
            df=clean_df(df)
            if df is None or df.empty or len(df)<30:
                time.sleep(2)
                continue
            return df
        except Exception as e:
            print(f"get_data {attempt+1}: {e}")
            time.sleep(3)
    return None

def get_spy_data(index):
    global SPY_CACHE
    if SPY_CACHE is None:
        try:
            spy = clean_df(yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False))
            if not spy.empty:
                SPY_CACHE = spy['Close']
        except: SPY_CACHE = None
    if SPY_CACHE is not None:
        try: return SPY_CACHE.reindex(index).ffill().bfill()
        except: return None
    return None

def add_features(df, ticker):
    df['MA20']=df['Close'].rolling(20).mean()
    df['MA50']=df['Close'].rolling(50).mean()
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
    spy_series = get_spy_data(df.index)
    df['SPY']= spy_series if spy_series is not None else df['Close']
    sent, news, analyst = get_finnhub_data(ticker)
    df['News_Score']=sent
    df['Analyst_Score']=analyst
    df['SPY_Trend']=df['SPY'].pct_change(20)
    return df

def prepare_dataset(df_raw, ticker, horizon_key):
    df=add_features(df_raw.copy(), ticker)
    cfg=HORIZONS[horizon_key]
    df['Future']=df['Close'].shift(-cfg["shift"])/df['Close']-1
    # DÜZƏLİŞ: thr çox yüksək idi, hamı GÖZLƏ olurdu
    thr=0.003 if horizon_key=="1s" else 0.008 if horizon_key=="1g" else 0.015 if horizon_key=="3g" else 0.02
    def etiket(x):
        if x>thr: return 2
        elif x<-thr: return 0
        else: return 1
    df['Label']=df['Future'].apply(etiket)
    df.dropna(inplace=True)
    features=['Close','MA20','MA50','RSI','MACD','BB_Pos','Volatility','Volume_Ratio','SPY','News_Score','Analyst_Score','SPY_Trend']
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

def train_model(ticker, horizon_key, X, y, df_info):
    if not HAS_TF: return None,None,None,None
    df, features, scaler = df_info
    brain_file=f"{DATA_PATH}/brain_{ticker}_{horizon_key}.keras"
    scaler_file=f"{DATA_PATH}/scaler_{ticker}_{horizon_key}.pkl"

    model=None
    if os.path.exists(brain_file):
        try:
            model=load_model(brain_file)
            print(f"🧠 Öz beyni yükləndi: {ticker} {horizon_key}")
        except: model=None
    
    if model is None:
        tsla_brain = f"{DATA_PATH}/brain_TSLA_{horizon_key}.keras"
        if ticker != "TSLA" and os.path.exists(tsla_brain):
            try:
                model=load_model(tsla_brain)
                print(f"🔄 Transfer: TSLA {horizon_key} -> {ticker}")
            except: model=None
    
    if model is None:
        print(f"📚 Sıfırdan: {ticker} {horizon_key}")
        model=Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(64, return_sequences=True), Dropout(0.25), LSTM(32), Dropout(0.25), Dense(16, activation='relu'), Dense(3, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Balanced class weight - GÖZLƏ çoxluğunu düzəldir
    try:
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {int(c): float(w) for c,w in zip(classes, weights)}
    except:
        class_weight = None

    model.fit(X,y,epochs=10,batch_size=32,verbose=0, class_weight=class_weight)
    model.save(brain_file)
    # Scaler-i də saxla!
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Saxlandı: {brain_file} + {scaler_file}")
    return model,scaler,features,df

def predict_horizon(ticker, horizon_key):
    df_raw=get_data(ticker,horizon_key)
    if df_raw is None: return None
    prepared=prepare_dataset(df_raw,ticker,horizon_key)
    if prepared[0] is None: return None
    X,y,df_info=prepared
    df,features,scaler_saved=df_info
    
    brain_file=f"{DATA_PATH}/brain_{ticker}_{horizon_key}.keras"
    scaler_file=f"{DATA_PATH}/scaler_{ticker}_{horizon_key}.pkl"
    
    # Scaler-i diskdən yükləməyə çalış, yoxdursa indiki scaler-i işlət
    scaler_to_use = scaler_saved
    if os.path.exists(scaler_file):
        try:
            with open(scaler_file, 'rb') as f:
                scaler_to_use = pickle.load(f)
        except: pass

    if not os.path.exists(brain_file):
        if HAS_TF:
            train_model(ticker,horizon_key,X,y,df_info)
        else: return None
    try:
        model=load_model(brain_file)
    except:
        if HAS_TF:
            model,scaler_to_use,_,_=train_model(ticker,horizon_key,X,y,df_info)
        else: return None

    lookback=30 if horizon_key=="1s" else 60
    try:
        last_df = df[features].apply(pd.to_numeric, errors='coerce').dropna().tail(lookback)
        if len(last_df) < lookback: return None
        last_n=scaler_to_use.transform(last_df)
    except Exception as e:
        print(f"scaler xətası {e}")
        return None

    X_pred=np.array([last_n])
    probs=model.predict(X_pred,verbose=0)[0]
    pred=int(np.argmax(probs)); conf=float(np.max(probs)*100)
    mapping={0:"SAT",1:"GÖZLƏ",2:"AL"}
    last=df.iloc[-1].to_dict()
    return mapping[pred],conf,probs,last

def log_prediction(ticker,horizon_key,decision,conf,price,probs):
    if price is None or price == 0:
        print(f"⚠ Qiymət 0, log yazılmır: {ticker} {horizon_key}")
        return
    now=datetime.now(BAKU_TZ)
    cfg=HORIZONS[horizon_key]
    target_time=now+timedelta(hours=1) if horizon_key=="1s" else now+timedelta(days=cfg["days"])
    row={"Date":now.strftime("%Y-%m-%d %H:%M"), "TargetDate":target_time.strftime("%Y-%m-%d %H:%M"), "Ticker":ticker, "Horizon":horizon_key, "Decision":decision, "Confidence":f"{conf:.0f}%", "PriceAtPred":price, "Checked":False, "Correct":False, "PriceAtCheck":0.0, "ActualChange":0.0}
    df=pd.DataFrame([row])
    if os.path.exists(PREDICTION_LOG):
        df.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG, index=False)

def get_overall_stats():
    try:
        if not os.path.exists(PREDICTION_LOG): return "📊 İlk proqnozlar",0,0,0
        df=pd.read_csv(PREDICTION_LOG)
        if df.empty: return "📊 İlk proqnozlar",0,0,0
        total=len(df)
        if 'Checked' not in df.columns or df[df['Checked']==True].empty:
            return f"📊 {total} proqnoz, nəticə gözləyir",0,total,0
        checked=df[df['Checked']==True]
        correct=len(checked[checked['Correct']==True]) if 'Correct' in checked.columns else 0
        acc=correct/len(checked)*100 if len(checked)>0 else 0
        last10=checked.tail(10)
        last10_acc=len(last10[last10['Correct']==True])/len(last10)*100 if len(last10)>0 and 'Correct' in last10.columns else 0
        emoji="🏆" if acc>=70 else "✅" if acc>=60 else "⚖" if acc>=50 else "📉"
        return f"{emoji} Uğur: {acc:.0f}% ({correct}/{len(checked)}) | Son 10: {last10_acc:.0f}% | Cəmi: {total}",acc,len(checked),correct
    except Exception as e:
        return f"📊 Stats: {e}",0,0,0

def evaluate_past_predictions():
    try:
        if not os.path.exists(PREDICTION_LOG): return "📊 İlk dəfə"
        df=pd.read_csv(PREDICTION_LOG)
        if df.empty: return "📊 Log boş"
        if 'Checked' not in df.columns: df['Checked']=False
        if 'Correct' not in df.columns: df['Correct']=False
        if 'PriceAtCheck' not in df.columns: df['PriceAtCheck']=0.0
        if 'ActualChange' not in df.columns: df['ActualChange']=0.0
        
        now = datetime.now(BAKU_TZ)
        updated = False
        for idx, row in df.iterrows():
            if row.get('Checked', False): continue
            try:
                target_str = str(row.get('TargetDate',''))
                target_time = datetime.strptime(target_str, "%Y-%m-%d %H:%M")
                target_time = BAKU_TZ.localize(target_time)
                if now < target_time: continue
                ticker = row.get('Ticker','TSLA')
                price_at = float(row.get('PriceAtPred',0))
                if price_at==0: continue
                curr_df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
                curr_df = clean_df(curr_df)
                if curr_df.empty: continue
                curr_price = float(curr_df['Close'].iloc[-1])
                change = (curr_price - price_at)/price_at
                decision = str(row.get('Decision',''))
                correct = False
                if decision=="AL" and change>0.005: correct=True
                elif decision=="SAT" and change<-0.005: correct=True
                elif decision=="GÖZLƏ" and abs(change)<=0.015: correct=True
                df.at[idx, 'Checked']=True
                df.at[idx, 'Correct']=correct
                df.at[idx, 'PriceAtCheck']=curr_price
                df.at[idx, 'ActualChange']=change
                updated=True
            except: continue
        if updated:
            df.to_csv(PREDICTION_LOG, index=False)
        return get_overall_stats()[0]
    except Exception as e:
        import traceback; traceback.print_exc()
        return get_overall_stats()[0]

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return False
    try:
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"}
        r=requests.post(url,data=data,timeout=15)
        print(f"Telegram: {r.status_code}")
        return r.status_code==200
    except Exception as e:
        print(f"Telegram xətası: {e}"); return False

def format_report(all_results, eval_msg):
    market_msg,_,_=get_market_countdown()
    overall_msg,acc,total,correct=get_overall_stats()
    now_str=datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')
    lines=[f"🤖 <b>{BOT_NAME}</b> {now_str}", market_msg, overall_msg]
    if "gözləyir" in eval_msg.lower() or "saxlandı" in eval_msg.lower():
        lines.append(f"📝 {eval_msg}")
    lines.append("─"*20)
    for ticker in sorted(set([r[0] for r in all_results])):
        ticker_results=[r for r in all_results if r[0]==ticker]
        real_price = get_current_price(ticker)
        if real_price is None:
            real_price = ticker_results[0][5].get('Close',0) if ticker_results and isinstance(ticker_results[0][5], dict) else 0
        lines.append(f"📈 <b>{ticker} ${real_price:.2f}</b> (real-time)")
        try:
            _, news, _ = get_finnhub_data(ticker)
            lines.append(f"📰 {news}")
        except: pass
        for _,hk,dec,conf,probs,last in ticker_results:
            cfg=HORIZONS[hk]
            icon="🟢" if dec=="AL" else "🔴" if dec=="SAT" else "🟡"
            brain_file=f"{DATA_PATH}/brain_{ticker}_{hk}.keras"
            learned="🧠" if os.path.exists(brain_file) else "📚"
            lines.append(f"{icon} {cfg['label']}: <b>{dec}</b> {conf:.0f}% {learned}")
            try:
                lines.append(f"   └ AL:{float(probs[2])*100:.0f}% GÖZLƏ:{float(probs[1])*100:.0f}% SAT:{float(probs[0])*100:.0f}%")
            except:
                lines.append(f"   └ {dec} {conf:.0f}%")
        lines.append("")
    lines.append("─"*20)
    lines.append(f"🕐 Bakı {datetime.now(BAKU_TZ).strftime('%H:%M')} | NY {datetime.now(NY_TZ).strftime('%H:%M')}")
    return "\n".join(lines)

def run():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN / CHAT_ID yoxdur! Secrets-i yoxla")
        # Local test üçün davam et, amma GitHub-da fail olacaq
    print(f"🚀 {BOT_NAME} - {TICKERS} - SMART V3")
    msg, is_open, _ = get_market_countdown()
    print(msg)
    eval_msg=evaluate_past_predictions()
    print(eval_msg)

    if not is_open:
        print("😴 Bazar bağlıdır")
        now = datetime.now(BAKU_TZ)
        if now.weekday() < 5:
            _, _, delta = get_market_countdown()
            if delta.total_seconds() < 7200:
                txt = f"🤖 {BOT_NAME} {datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')}\n{msg}\n{eval_msg}\n⏳ Açılışa hazırlaşıram..."
                send_telegram(txt)
        return

    all_results=[]
    for ticker in TICKERS:
        for hk in ["1s","1g","3g","5g"]:
            if hk=="1s" and not is_us_market_open():
                print(f"⏭ {ticker} {hk} keçildi - bazar bağlı")
                continue
            try:
                res=predict_horizon(ticker,hk)
                if res is None: 
                    print(f"⚠ {ticker} {hk} alınmadı")
                    continue
                dec,conf,probs,last=res
                real_p = get_current_price(ticker)
                price_to_log = real_p if real_p and real_p>0 else last.get('Close',0)
                if price_to_log and price_to_log>0:
                    log_prediction(ticker,hk,dec,conf,price_to_log,probs)
                all_results.append((ticker,hk,dec,conf,probs,last))
                print(f"✅ {ticker} {hk}: {dec} {conf:.0f}% @ ${price_to_log:.2f}")
            except Exception as e:
                print(f"❌ {ticker} {hk}: {e}")
                import traceback; traceback.print_exc()
    if all_results:
        report=format_report(all_results, eval_msg)
        print("\n"+report)
        send_telegram(report)
    else:
        txt=f"🤖 {BOT_NAME} {datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')}\n{msg}\n⚠ Data alınmadı"
        send_telegram(txt)

if __name__=="__main__":
    try:
        run()
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                msg = f"🤖 {BOT_NAME} XƏTA {datetime.now(BAKU_TZ).strftime('%H:%M')}\n⚠ {str(e)[:500]}"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        except: pass
    finally:
        import sys; sys.exit(0)
