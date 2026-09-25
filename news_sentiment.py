import os, json, requests
from datetime import datetime, timedelta

CACHE_FILE = "news_cache.json"

def get_news_sentiment():
    # Cache yoxla - bu gün üçün varsa API-yə getmə (EKONOM)
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if cache.get("date") == today:
                    # Yeni xəbər yoxdur, amma köhnəni qaytarırıq
                    return cache.get("sentiment", 0), cache.get("headline", "yeni xəbər yoxdur"), False
        except:
            pass

    key = os.getenv("FINNHUB_KEY")
    if not key:
        return 0, "yeni xəbər yoxdur", False

    try:
        # Yalnız dünən-bugün, TSLA üçün
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol=TSLA&from={yesterday}&to={today}&token={key}"
        r = requests.get(url, timeout=10)
        if r.status_code == 429: # Limit bitib
            return 0, "yeni xəbər yoxdur", False
        data = r.json()[:20] # yalnız 20 xəbər

        # YALNIZ TSLA/Tesla sözü olanlar
        filtered = [n for n in data if "tesla" in (n.get("headline","")+n.get("summary","")).lower() or "tsla" in (n.get("headline","")).lower()]
        if not filtered:
            return 0, "yeni xəbər yoxdur", False

        # VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
        except:
            from nltk.sentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()

        # Köhnə başlıqları oxu, yeniləri tap
        old_headlines = []
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    old_headlines = json.load(f).get("headlines", [])
            except:
                pass

        new_items = [n for n in filtered if n.get("headline") not in old_headlines]
        if not new_items: # YENİ xəbər yoxdur - reaksiya vermə!
            return 0, "yeni xəbər yoxdur", False

        # Yalnız YENİ xəbərlərə skor
        scores = [analyzer.polarity_scores(n.get("headline",""))["compound"] for n in new_items]
        sentiment = sum(scores)/len(scores) if scores else 0
        headline = new_items[0].get("headline","")[:80]

        # Cache-i yenilə
        all_headlines = [n.get("headline") for n in filtered]
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "sentiment": sentiment, "headline": headline, "headlines": all_headlines}, f, ensure_ascii=False)

        return sentiment, headline, True

    except Exception as e:
        print(f"News error: {e}")
        return 0, "yeni xəbər yoxdur", False
