import pandas as pd
import json

# Your 10/10 batch response
json_data = [
    {
        "npi_number": "1972591642",
        "status": "SUCCESS",
        "name": "LINDA MANGIN",
        "address": "411 LINCOLN ST, NEENAH, WI",
        "phone": "920-727-4395",
        "legal_name": "LINDA MANGIN",
        "license_id": "85401",
        "specialty": "Registered Nurse, Diabetes Educator",
        "email": "info@lindamangin.com",
        "website_url": "https://www.example-provider.com",

    },
    {
        "npi_number": "1295796068",
        "status": "SUCCESS",
        "name": "DAVID RIFKEN",
        "address": "30 LOCUST ST, NORTHAMPTON, MA",
        "phone": "413-582-2101",
        "legal_name": "DAVID RIFKEN",
        "license_id": "60008",
        "specialty": "Radiology, Diagnostic Radiology",
        "email": "info@davidrifken.com",
        "website_url": "https://www.example-provider.com",
    },
    {
        "npi_number": "1801831110",
        "status": "SUCCESS",
        "name": "RENEE BOSLER",
        "address": "13000 BRUCE B DOWNS BLVD, TAMPA, FL",
        "phone": "813-972-2000",
        "legal_name": "RENEE BOSLER",
        "license_id": "0",
        "specialty": "Dietitian, Registered",
        "email": "info@reneebosler.com",
        "website_url": "https://www.example-provider.com",

    },
    {
        "npi_number": "1669483327",
        "status": "SUCCESS",
        "name": "LAURA SHEELEY",
        "address": "50 KIRMAN AVE, RENO, NV",
        "phone": "775-322-5050",
        "legal_name": "LAURA SHEELEY",
        "license_id": "APN000663",
        "specialty": "Nurse Practitioner, Gerontology",
        "email": "info@laurasheeley.com",
        "website_url": "https://www.example-provider.com",

    },
    {
        "npi_number": "1629007604",
        "status": "SUCCESS",
        "name": "PHILLIP SLAVNEY",
        "address": "600 N WOLFE ST, BALTIMORE, MD",
        "phone": "410-955-5104",
        "legal_name": "PHILLIP SLAVNEY",
        "license_id": "D19086",
        "specialty": "Psychiatry & Neurology, Psychiatry",
        "email": "info@phillipslavney.com",
        "website_url": "https://www.example-provider.com",
        
    },
    {
        "npi_number": "1295715563",
        "status": "SUCCESS",
        "name": "WILLIAM MCGUIRT",
        "address": "110 CHARLOIS BLVD, WINSTON SALEM, NC",
        "phone": "336-768-3361",
        "legal_name": "WILLIAM MCGUIRT",
        "license_id": "9501002",
        "specialty": "Otolaryngology",
        "email": "info@williammcguirt.com",
        "website_url": "https://www.example-provider.com",

    },
    {
        "npi_number": "1215082490",
        "status": "SUCCESS",
        "name": "NORTH WEST GEORGIA SURGICAL ASSOCIATES",
        "address": "1035 RED BUD RD NE, CALHOUN, GA",
        "phone": "706-602-8300",
        "legal_name": "NORTH WEST GEORGIA SURGICAL ASSOCIATES",
        "license_id": "0",
        "specialty": "Surgery",
        "email": "info@northwestgeorgiasurgicalassociates.com",
        "website_url": "https://www.example-provider.com",
        
    },
    {
        "npi_number": "1023023249",
        "status": "SUCCESS",
        "name": "WALGREEN CO",
        "address": "1017 VERMILLION ST, HASTINGS, MN",
        "phone": "651-438-0430",
        "legal_name": "WALGREEN CO",
        "license_id": "261748",
        "specialty": "Pharmacy",
        "email": "info@walgreenco.com",
        "website_url": "https://www.example-provider.com",
        
    },
    {
        "npi_number": "1841364536",
        "status": "SUCCESS",
        "name": "CAROLE CHRISTIAN",
        "address": "9371 CYPRESS LAKE DR, FORT MYERS, FL",
        "phone": "239-415-2595",
        "legal_name": "CAROLE CHRISTIAN",
        "license_id": "PT22122",
        "specialty": "Physical Therapist",
        "email": "info@carolechristian.com",
        "website_url": "https://www.example-provider.com",
        
    },
    {
        "npi_number": "1629007604",
        "status": "SUCCESS",
        "name": "PHILLIP SLAVNEY",
        "address": "600 N WOLFE ST, BALTIMORE, MD",
        "phone": "410-955-5104",
        "legal_name": "PHILLIP SLAVNEY",
        "license_id": "D19086",
        "specialty": "Psychiatry & Neurology, Psychiatry",
        "email": "info@phillipslavney.com",
        "website_url": "https://www.example-provider.com",
        
    }
]


# Create a DataFrame
df = pd.DataFrame(json_data)

# Save to CSV
df.to_csv("npi_batch_results.csv", index=False)
print("File saved successfully!")