"""
NEWS SENTIMENT - V6.7 FINAL - FINNHUB + VADER + CACHE
"""
import os, json, requests
from datetime import datetime, timedelta

CACHE_FILE = "news_cache.json"
CACHE_HOURS = 6

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def get_news_sentiment(ticker="TSLA"):
    """
    Həm get_news_sentiment() həm də get_news_sentiment("TSLA") işləyir
    Qayıdır: (sentiment_score, headline, is_new)
    """
    # 1. Cache yoxla
    cache = _load_cache()
    now = datetime.utcnow()
    if ticker in cache:
        try:
            cached = cache[ticker]
            cached_time = datetime.fromisoformat(cached.get("time", "2000-01-01"))
            if (now - cached_time) < timedelta(hours=CACHE_HOURS):
                print(f"📰 News cache OK: {cached.get('headline','')[:50]}")
                return cached["sentiment"], cached["headline"], False
        except:
            pass

    # 2. Finnhub-dan çək
    api_key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    headline = "yeni xəbər yoxdur"
    sentiment = 0
    is_new = False

    if not api_key:
        print("⚠️ FINNHUB_API_KEY yoxdur - xəbər alınmadı")
        return sentiment, headline, is_new

    try:
        # Son 3 günün xəbərləri
        to_date = now.strftime("%Y-%m-%d")
        from_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if not data or not isinstance(data, list):
            print(f"📰 {ticker} xəbər boş gəldi")
            return sentiment, headline, is_new

        # Ən yeni 1 xəbər
        latest = data[0]
        headline = latest.get("headline", "yeni xəbər yoxdur")[:200]
        summary = latest.get("summary", "") or headline

        # VADER ilə analiz
        if VADER_AVAILABLE:
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(summary)
            sentiment = round(vs["compound"] * 100) # -100..+100
        else:
            # sadə keyword
            sentiment = 10 if "up" in summary.lower() or "buy" in summary.lower() else -10 if "down" in summary.lower() else 0

        print(f"📰 NEW {ticker}: {sentiment} | {headline[:80]}")
        is_new = True

        # Cache-ə yaz
        cache[ticker] = {
            "time": now.isoformat(),
            "sentiment": sentiment,
            "headline": headline
        }
        _save_cache(cache)

    except Exception as e:
        print(f"📰 News error {ticker}: {e}")
        # cache-də varsa köhnəni qaytar
        if ticker in cache:
            return cache[ticker].get("sentiment",0), cache[ticker].get("headline","yeni xəbər yoxdur"), False

    return sentiment, headline, is_new

# Köhnə çağırışlar üçün backward compatibility
def get_news_sentiment_noarg():
    return get_news_sentiment("TSLA")
