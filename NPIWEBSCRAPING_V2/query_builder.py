def build_search_queries(npi_provider: dict) -> list[str]:
    basic = npi_provider.get("basic", {})

    name = " ".join(
        filter(
            None,
            [
                basic.get("first_name"),
                basic.get("middle_name"),
                basic.get("last_name"),
                basic.get("credential"),
            ],
        )
    )

    taxonomy = ""
    for t in npi_provider.get("taxonomies", []):
        if t.get("primary"):
            taxonomy = t.get("desc", "")
            break

    location_address = next(
        (a for a in npi_provider.get("addresses", []) if a.get("address_purpose") == "LOCATION"),
        {}
    )

    city = location_address.get("city", "")
    state = location_address.get("state", "")

    return [
        f"{name} {taxonomy} {city} {state}",
        f"{name} {city} {taxonomy}",
        f"{name} medical practice {city}"
    ]
