import os
from typing import Any
from dotenv import load_dotenv
from ddgs import DDGS
import feedparser

load_dotenv()


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Try Tavily first (if configured), fallback to DuckDuckGo."""

    # Smart query enhancement for news
    lower_q = query.lower()
    if "news" in lower_q or "headline" in lower_q:
        query = (
            f"{query} site:bbc.com OR site:reuters.com "
            "OR site:cnn.com OR site:ndtv.com OR site:thehindu.com"
        )

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    # 🔹 Try Tavily first
    if tavily_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            result = client.search(query=query, max_results=max_results)

            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score"),
                }
                for item in result.get("results", [])
            ]
        except Exception as e:
            print("Tavily failed, falling back to DuckDuckGo:", e)

    # Fallback to DuckDuckGo
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))

            return [
                {
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "content": row.get("body", ""),
                }
                for row in rows
            ]
    except Exception as e:
        print("DDGS search failed:", e)
        return []
    
def get_latest_news(max_results=5):
    feeds = [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "http://rss.cnn.com/rss/edition.rss",
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://www.thehindu.com/news/feeder/default.rss"
    ]

    results = []

    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:max_results]:
            results.append({
                "title": entry.title,
                "url": entry.link,
                "published": entry.get("published", ""),
            })

    return results[:max_results]