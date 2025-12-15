import urllib.parse
from ddgs import DDGS

SEARCH_RESULTS_LIMIT = 5  # limit to top results

def search_provider_urls(query: str) -> list[str]:
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=SEARCH_RESULTS_LIMIT)
            for r in results:
                href = r.get("href")
                if href and href.startswith("http"):
                    urls.append(href)
    except Exception:
        pass
    return urls
