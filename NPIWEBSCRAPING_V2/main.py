import json
import asyncio
from orchestrator import scrape_provider

INPUT_FILE = "npi_input.json"         # input JSON with 200 providers
OUTPUT_FILE = "scraped_results.json"  # output JSON file
MAX_CONCURRENT_BROWSERS = 4           # adjust for your machine

async def main():
    # Load input JSON
    with open(INPUT_FILE, "r") as f:
        npi_response = json.load(f)

    providers = npi_response.get("results", [])
    sem = asyncio.Semaphore(MAX_CONCURRENT_BROWSERS)

    # Worker function with semaphore
    async def worker(provider):
        async with sem:
            data = await scrape_provider(provider)
            return {
                "npi": provider.get("number"),
                "scraped_data": data.model_dump()
            }

    # Run all providers concurrently
    results = await asyncio.gather(*[worker(p) for p in providers])

    # Print JSON to console
    print(json.dumps(results, indent=2))

    # Write JSON to a single file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
