import os, json
from datetime import datetime, timedelta

CACHE_FILE = "news_cache.json"

def get_news_sentiment():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        key = os.getenv("FINNHUB_KEY")
        if not key:
            print("FINNHUB_KEY yoxdur -> yeni xəbər yoxdur")
            return 0, "yeni xəbər yoxdur", False

        # Cache check
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    if cache.get("date") == today:
                        return cache.get("sentiment",0), cache.get("headline","yeni xəbər yoxdur"), False
            except:
                pass

        import requests
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol=TSLA&from={yesterday}&to={today}&token={key}"
        r = requests.get(url, timeout=10)
        if r.status_code!= 200:
            return 0, "yeni xəbər yoxdur", False

        data = r.json()[:10]
        filtered = [n for n in data if "tesla" in (n.get("headline","").lower()) or "tsla" in (n.get("headline","").lower())]
        if not filtered:
            return 0, "yeni xəbər yoxdur", False

        # Sentiment - VADER yoxdursa sadə hesab
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(n.get("headline",""))["compound"] for n in filtered[:3]]
            sentiment = sum(scores)/len(scores) if scores else 0
        except:
            sentiment = 0

        headline = filtered[0].get("headline","")[:80]

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "sentiment": sentiment, "headline": headline, "headlines": [n.get("headline") for n in filtered]}, f, ensure_ascii=False)

        # Həmişə YENİ kimi saymayaq - cache olduğuna görə artıq False qaytaracaq sabah
        # İlk dəfə True
        return sentiment, headline, True

    except Exception as e:
        print(f"News error (çökmədi): {e}")
        return 0, "yeni xəbər yoxdur", False
