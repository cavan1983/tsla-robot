import os, json, requests
from datetime import datetime, timedelta
CACHE_FILE = "news_cache.json"
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = True
except:
    VADER = False

def _load():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: return json.load(f)
        except: return {}
    return {}
def _save(c):
    try:
        with open(CACHE_FILE,"w") as f: json.dump(c,f,indent=2)
    except: pass

def get_news_sentiment(ticker="TSLA"):
    cache=_load()
    now=datetime.utcnow()
    # cache 6 saat
    if ticker in cache:
        try:
            ct=datetime.fromisoformat(cache[ticker]["time"])
            if now-ct < timedelta(hours=6):
                return cache[ticker]["sentiment"], cache[ticker]["headline"], False
        except: pass

    key=os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    if not key:
        print("FINNHUB KEY yoxdur")
        return 0, "yeni xəbər yoxdur", False

    try:
        to_d=now.strftime("%Y-%m-%d")
        from_d=(now-timedelta(days=7)).strftime("%Y-%m-%d")
        url=f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_d}&to={to_d}&token={key}"
        print(f"Finnhub sorğu: {ticker}")
        r=requests.get(url,timeout=15)
        data=r.json()
        if not data:
            return 0, "yeni xəbər yoxdur", False

        # Ən yeni xəbər
        news=data[0]
        headline=news.get("headline","yeni xəbər yoxdur")[:250]
        summary=news.get("summary","") or headline

        if VADER:
            analyzer=SentimentIntensityAnalyzer()
            score=analyzer.polarity_scores(summary)["compound"]
            sentiment=int(score*100)
        else:
            sentiment=0

        print(f"📰 NEW: {sentiment} | {headline[:100]}")
        cache[ticker]={"time":now.isoformat(),"sentiment":sentiment,"headline":headline}
        _save(cache)
        return sentiment, headline, True

    except Exception as e:
        print(f"News error: {e}")
        if ticker in cache:
            return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
        return 0, "yeni xəbər yoxdur", False
