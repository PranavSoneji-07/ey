from bs4 import BeautifulSoup
from models import ScrapedProviderData
from utils import PHONE_REGEX


def extract_provider_data(html: str, url: str) -> ScrapedProviderData:
    soup = BeautifulSoup(html, "html.parser")

    visible_text = soup.get_text(" ", strip=True)

    phones = set(PHONE_REGEX.findall(visible_text))
    addresses = set()

    # tel: links
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("tel:"):
            phones.add(a["href"].replace("tel:", "").strip())

    # Address heuristics
    for tag in soup.find_all(["p", "div", "span", "li"]):
        text = tag.get_text(" ", strip=True)
        if any(x in text for x in [" St", " Rd", " Ave", " Blvd", " Drive", " Way"]):
            if any(c.isdigit() for c in text):
                addresses.add(text)

    title = soup.title.string.strip() if soup.title else None

    return ScrapedProviderData(
        name=title,
        phone_numbers=list(phones),
        addresses=list(addresses),
        source_urls=[url],
    )
