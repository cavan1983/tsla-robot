
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING'] = '0'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import time, requests, json, glob
import pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8909079473:AAHX7XMiGwou8sbd5Rs1B-0CU0LBUgHc32w")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1434358288")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "dajgcb1r01qhhp593b1gdajgcb1r01qhhp593b20")
TICKERS = ["TSLA"]

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

FINNHUB_CACHE_FILE = f"{DATA_PATH}/finnhub_cache.json"
PREDICTION_LOG = f"{DATA_PATH}/predictions.csv"

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
        return f"⏰ Açılışa {h}s {m}dəq (həftəsonu)", False, d
    today_open = now.replace(hour=17, minute=30, second=0, microsecond=0)
    today_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if now < today_open:
        d = today_open-now
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        return f"⏰ Açılışa {h}s {m}dəq qaldı", False, d
    else:
        d = today_close-now
        h = int(d.total_seconds()//3600); m = int((d.total_seconds()%3600)//60)
        return f"🟢 Bazar AÇIQ | Bağlanmaya {h}s {m}dəq", True, d

def is_us_market_open():
    _, is_open, _ = get_market_countdown()
    return is_open

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
            print(f"get_data cəhd {attempt+1} xətası: {e}")
            time.sleep(3)
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
    try:
        spy=clean_df(yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False))
        df['SPY']=spy['Close'].reindex(df.index).ffill().bfill()
    except:
        df['SPY']=df['Close']
    sent, news, analyst = get_finnhub_data(ticker)
    df['News_Score']=sent
    df['Analyst_Score']=analyst
    df['SPY_Trend']=df['SPY'].pct_change(20)
    return df

def prepare_dataset(df_raw, ticker, horizon_key):
    df=add_features(df_raw.copy(), ticker)
    cfg=HORIZONS[horizon_key]
    df['Future']=df['Close'].shift(-cfg["shift"])/df['Close']-1
    thr=0.015 if horizon_key=="1s" else 0.02 if horizon_key=="1g" else 0.025 if horizon_key=="3g" else 0.03
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
    h5_file=f"{DATA_PATH}/brain_{ticker}_{horizon_key}.h5"
    # köhnə h5 varsa sil, keras istifadə et
    if os.path.exists(h5_file) and not os.path.exists(brain_file):
        try: os.remove(h5_file)
        except: pass
    model=None
    if os.path.exists(brain_file):
        try:
            model=load_model(brain_file)
        except:
            model=None
    if model is None:
        model=Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(64, return_sequences=True), Dropout(0.25), LSTM(32), Dropout(0.25), Dense(16, activation='relu'), Dense(3, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X,y,epochs=8,batch_size=32,verbose=0)
    model.save(brain_file)
    return model,scaler,features,df

def predict_horizon(ticker, horizon_key):
    df_raw=get_data(ticker,horizon_key)
    if df_raw is None: return None
    prepared=prepare_dataset(df_raw,ticker,horizon_key)
    if prepared[0] is None: return None
    X,y,df_info=prepared
    df,features,scaler=df_info
    brain_file=f"{DATA_PATH}/brain_{ticker}_{horizon_key}.keras"
    h5_file=f"{DATA_PATH}/brain_{ticker}_{horizon_key}.h5"
    # əgər model yoxdursa öyrət
    if not os.path.exists(brain_file) and not os.path.exists(h5_file):
        if HAS_TF:
            train_model(ticker,horizon_key,X,y,df_info)
        else:
            return None
    try:
        if os.path.exists(brain_file):
            model=load_model(brain_file)
        else:
            model=load_model(h5_file)
    except:
        if HAS_TF:
            model,_,_,_=train_model(ticker,horizon_key,X,y,df_info)
        else:
            return None
    lookback=30 if horizon_key=="1s" else 60
    try:
        last_n=scaler.transform(df[features].apply(pd.to_numeric, errors='coerce').dropna().tail(lookback))
    except:
        return None
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
        if not os.path.exists(PREDICTION_LOG): return "📊 İlk proqnozlar - öyrənirəm",0,0,0
        df=pd.read_csv(PREDICTION_LOG)
        if df.empty: return "📊 İlk proqnozlar - öyrənirəm",0,0,0
        total=len(df)
        # Hələ yoxlanmamış
        if 'Checked' not in df.columns or df[df['Checked']==True].empty:
            return f"📊 {total} proqnoz saxlandı, nəticə 1-5 gündə",0,total,0
        checked=df[df['Checked']==True]
        correct=len(checked[checked['Correct']==True]) if 'Correct' in checked.columns else 0
        acc=correct/len(checked)*100 if len(checked)>0 else 0
        last10=checked.tail(10)
        last10_acc=len(last10[last10['Correct']==True])/len(last10)*100 if len(last10)>0 and 'Correct' in last10.columns else 0
        if acc>=70: emoji="🏆"
        elif acc>=60: emoji="✅"
        elif acc>=50: emoji="⚖️"
        else: emoji="📉"
        return f"{emoji} Uğur: {acc:.0f}% ({correct}/{len(checked)}) | Son 10: {last10_acc:.0f}% | Cəmi: {total}",acc,len(checked),correct
    except Exception as e:
        return f"📊 Stats: {e}",0,0,0

def evaluate_past_predictions():
    # Keçmiş proqnozları yoxla
    try:
        if not os.path.exists(PREDICTION_LOG):
            return "📊 İlk dəfə işləyir"
        df=pd.read_csv(PREDICTION_LOG)
        if df.empty: return "📊 Log boşdur"
        # Sadə yoxlama
        unchecked = df[df['Checked']==False] if 'Checked' in df.columns else df
        if unchecked.empty:
            return get_overall_stats()[0]
        # Qiymət yoxlaması (sadələşdirilmiş - GitHub-da Drive yoxdur)
        return get_overall_stats()[0]
    except Exception as e:
        return f"📊 Yoxlama: {e}"

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return False
    try:
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"}
        r=requests.post(url,data=data,timeout=15)
        print(f"Telegram: {r.status_code}")
        if r.status_code!=200:
            print(r.text)
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
        last_price=ticker_results[0][3].get('Close',0) if ticker_results else 0
        lines.append(f"📈 <b>{ticker} ${last_price:.2f}</b>")
        # Finnhub news
        try:
            _, news, _ = get_finnhub_data(ticker)
            lines.append(f"📰 {news}")
        except: pass
        for _,hk,dec,conf,probs,last in ticker_results:
            cfg=HORIZONS[hk]
            icon="🟢" if dec=="AL" else "🔴" if dec=="SAT" else "🟡"
            # Öyrənmə faizi - confidence ilə
            brain_file=f"{DATA_PATH}/brain_{ticker}_{hk}.keras"
            learned="🧠" if os.path.exists(brain_file) else "📚"
            lines.append(f"{icon} {cfg['label']}: <b>{dec}</b> {conf:.0f}% {learned}")
            # SAT/AL ehtimalları
            lines.append(f"   └ AL:{probs[2]*100:.0f}% GÖZLƏ:{probs[1]*100:.0f}% SAT:{probs[0]*100:.0f}%")
        lines.append("")
    lines.append("─"*20)
    lines.append(f"🕐 Bakı {datetime.now(BAKU_TZ).strftime('%H:%M')} | NY {(datetime.now(BAKU_TZ)-timedelta(hours=8)).strftime('%H:%M')}")
    lines.append(f"⏳ Növbəti 1 saatdan sonra")
    return "\n".join(lines)

def run():
    print(f"🚀 {BOT_NAME} - {TICKERS}")
    msg,_,_=get_market_countdown()
    print(msg)
    eval_msg=evaluate_past_predictions()
    print(eval_msg)

    all_results=[]
    for ticker in TICKERS:
        for hk in ["1s","1g","3g","5g"]:
            if hk=="1s" and not is_us_market_open():
                print(f"⏭️ {ticker} {hk} keçildi - bazar bağlı")
                continue
            try:
                res=predict_horizon(ticker,hk)
                if res is None: 
                    print(f"⚠️ {ticker} {hk} alınmadı")
                    continue
                dec,conf,probs,last=res
                log_prediction(ticker,hk,dec,conf,last.get('Close',0),probs)
                all_results.append((ticker,hk,dec,conf,probs,last))
                print(f"✅ {ticker} {hk}: {dec} {conf:.0f}% @ ${last.get('Close',0):.2f}")
            except Exception as e:
                print(f"❌ {ticker} {hk}: {e}")
                import traceback; traceback.print_exc()
    if all_results:
        report=format_report(all_results, eval_msg)
        print("\n"+report)
        send_telegram(report)
    else:
        txt=f"🤖 {BOT_NAME} {datetime.now(BAKU_TZ).strftime('%d.%m %H:%M')}\n{msg}\n⚠️ Data alınmadı"
        send_telegram(txt)

if __name__=="__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"💥 Kritik xəta: {e}")
        traceback.print_exc()
        # Telegram-a xəta mesajı göndər, amma exit 1 etmə
        try:
            import os, requests
            from datetime import datetime
            import pytz
            BAKU_TZ = pytz.timezone("Asia/Baku")
            BOT_NAME = "Cavanshir83Bot"
            TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8909079473:AAHX7XMiGwou8sbd5Rs1B-0CU0LBUgHc32w")
            TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1434358288")
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                msg = f"🤖 {BOT_NAME} XƏTA {datetime.now(BAKU_TZ).strftime('%H:%M')}\n⚠️ {str(e)[:500]}\n🔄 Növbəti saatda yenidən cəhd"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        except:
            pass
    finally:
        import sys
        sys.exit(0)

