import os, json, requests
from datetime import datetime, timedelta

# DÜZƏLDİLDİ: hamısı data/ içində
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_FILE = os.path.join(DATA_DIR, "news_cache.json")
CACHE_HOURS = 6

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = True
except:
    VADER = False

def _load():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save(cache):
    try:
        with open(CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except:
        pass

def get_news_sentiment(ticker="KO"):
    cache = _load()
    now = datetime.utcnow()

    if ticker in cache:
        try:
            ct = datetime.fromisoformat(cache[ticker].get("time", "2000-01-01"))
            age_hours = (now - ct).total_seconds() / 3600
            if age_hours < CACHE_HOURS:
                print(f"📰 Cache TƏZƏDİR ({age_hours:.1f} saat) - API işləmir")
                return cache[ticker].get("sentiment", 0), cache[ticker].get("headline", "yeni xəbər yoxdur"), False
        except:
            pass

    print(f"📰 Cache köhnədir - Finnhub-a 1 sorğu gedir {ticker}...")
    key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    if not key:
        print("⚠️ FINNHUB_API_KEY yoxdur")
        if ticker in cache:
            return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
        return 0, "yeni xəbər yoxdur", False

    try:
        to_d = now.strftime("%Y-%m-%d")
        from_d = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_d}&to={to_d}&token={key}"
        r = requests.get(url, timeout=15)

        if r.status_code == 429:
            print("🛑 FINNHUB LIMIT 429")
            if ticker in cache:
                return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
            return 0, "yeni xəbər yoxdur", False

        data = r.json()
        if not data or len(data) == 0:
            if ticker in cache:
                return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
            return 0, "yeni xəbər yoxdur", False

        latest = data[0]
        headline = latest.get("headline", "yeni xəbər yoxdur")[:250]
        summary = latest.get("summary", "") or headline

        if ticker in cache and cache[ticker].get("headline") == headline:
            print(f"📰 Eyni xəbər: {headline[:60]}")
            return cache[ticker].get("sentiment",0), headline, False

        if VADER:
            analyzer = SentimentIntensityAnalyzer()
            sentiment = int(analyzer.polarity_scores(summary)["compound"] * 100)
        else:
            sentiment = 0

        print(f"📰 YENİ XƏBƏR {ticker}: {sentiment} | {headline[:80]}")
        cache[ticker] = {"time": now.isoformat(), "sentiment": sentiment, "headline": headline}
        _save(cache)
        return sentiment, headline, True

    except Exception as e:
        print(f"📰 Xəbər error: {e}")
        if ticker in cache:
            return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
        return 0, "yeni xəbər yoxdur", False
