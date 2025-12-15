import re
from playwright.async_api import async_playwright
from models import ScrapedProviderData
from utils import PHONE_REGEX


async def fetch_with_browser(url: str) -> ScrapedProviderData:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        page = await context.new_page()

        try:
            await page.goto(url, timeout=25000, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)

            text = await page.inner_text("body")
            title = await page.title()

        except Exception:
            await browser.close()
            return ScrapedProviderData()

        await browser.close()

    phones = list(set(PHONE_REGEX.findall(text)))
    addresses = []

    for line in text.split("\n"):
        if any(x in line for x in [" St", " Rd", " Ave", " Blvd"]) and any(c.isdigit() for c in line):
            addresses.append(line.strip())

    return ScrapedProviderData(
        name=title,
        phone_numbers=phones[:3],
        addresses=addresses[:3],
        source_urls=[url],
    )
