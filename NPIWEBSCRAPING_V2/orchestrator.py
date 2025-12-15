from models import ScrapedProviderData
from query_builder import build_search_queries
from search import search_provider_urls
from browser_fetcher import fetch_with_browser

async def scrape_provider(npi_payload: dict) -> ScrapedProviderData:
    aggregated = ScrapedProviderData()

    # Seed phone numbers from NPI
    for addr in npi_payload.get("addresses", []):
        if addr.get("telephone_number"):
            aggregated.phone_numbers.append(addr["telephone_number"])

    # Fill taxonomies / profession from NPI
    aggregated.taxonomies = [t["desc"] for t in npi_payload.get("taxonomies", [])]
    if aggregated.taxonomies:
        aggregated.profession = aggregated.taxonomies[0]

    # Build search queries
    queries = build_search_queries(npi_payload)

    # Search and fetch pages
    for q in queries:
        urls = search_provider_urls(q)

        # Filter out generic directories
        skip_domains = ["npino.com", "zocdoc.com", "healthgrades.com", "vitals.com"]
        urls = [u for u in urls if not any(bad in u for bad in skip_domains)]

        for url in urls:
            data = await fetch_with_browser(url)

            aggregated.phone_numbers.extend(data.phone_numbers)
            aggregated.addresses.extend(data.addresses)
            aggregated.source_urls.extend(data.source_urls)

            # Stop early if useful data found
            if data.addresses or data.phone_numbers:
                break

    # Deduplicate
    aggregated.phone_numbers = list(set(aggregated.phone_numbers))
    aggregated.addresses = list(set(aggregated.addresses))
    aggregated.source_urls = list(set(aggregated.source_urls))

    # Ensure name is filled
    if not aggregated.name:
        basic = npi_payload.get("basic", {})
        aggregated.name = " ".join(filter(None, [
            basic.get("first_name"),
            basic.get("middle_name"),
            basic.get("last_name"),
            basic.get("credential")
        ]))

    return aggregated
