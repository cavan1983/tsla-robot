import os, json, requests
from datetime import datetime, timedelta

CACHE_FILE = "news_cache.json"
CACHE_HOURS = 6 # 6 saatdan bir yoxla, tez-tez yox

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = True
except:
    VADER = False

def _load():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def get_news_sentiment(ticker="TSLA"):
    """
    LIMIT QORUYUCU:
    - Cache təzədirsə (6 saat) -> API-yə toxunmur, cache-dən verir
    - Cache köhnədirsə -> 1 dəfə API-yə vurur
    - Limit bitibsə -> cache-dən köhnəni qaytarır
    Return: (sentiment, headline, is_new)
    """
    cache = _load()
    now = datetime.utcnow()

    # 1. CACHE TƏZƏDİRSƏ - API-yə VURMA!
    if ticker in cache:
        try:
            ct = datetime.fromisoformat(cache[ticker].get("time", "2000-01-01"))
            age_hours = (now - ct).total_seconds() / 3600
            if age_hours < CACHE_HOURS:
                print(f"📰 Cache TƏZƏDİR ({age_hours:.1f} saat) - API işləmir, limit qorundu")
                return cache[ticker].get("sentiment", 0), cache[ticker].get("headline", "yeni xəbər yoxdur"), False
        except:
            pass

    # 2. CACHE KÖHNƏDİR - İNDİ API-yə VUR
    print(f"📰 Cache köhnədir - Finnhub-a 1 sorğu gedir...")
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

        # Limit bitib?
        if r.status_code == 429:
            print("🛑 FINNHUB LIMIT 429 - cache-dən qaytarıram")
            if ticker in cache:
                return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
            return 0, "yeni xəbər yoxdur", False

        data = r.json()
        if not data or len(data) == 0:
            print("📰 Xəbər boş gəldi")
            if ticker in cache:
                return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
            return 0, "yeni xəbər yoxdur", False

        latest = data[0]
        headline = latest.get("headline", "yeni xəbər yoxdur")[:250]
        summary = latest.get("summary", "") or headline

        # Eyni xəbərdirsə, yeni deyil
        if ticker in cache and cache[ticker].get("headline") == headline:
            print(f"📰 Eyni xəbərdir, yeni deyil: {headline[:60]}")
            return cache[ticker].get("sentiment",0), headline, False

        # Yeni xəbərdir - sentiment hesabla
        if VADER:
            analyzer = SentimentIntensityAnalyzer()
            sentiment = int(analyzer.polarity_scores(summary)["compound"] * 100)
        else:
            sentiment = 0

        print(f"📰 YENİ XƏBƏR TAPILDI: {sentiment} | {headline[:80]}")
        cache[ticker] = {"time": now.isoformat(), "sentiment": sentiment, "headline": headline}
        _save(cache)
        return sentiment, headline, True

    except Exception as e:
        print(f"📰 Xəbər error: {e}")
        if ticker in cache:
            return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False
        return 0, "yeni xəbər yoxdur", False
