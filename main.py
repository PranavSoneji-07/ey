import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional
import httpx 
import asyncio

import shutil
import pandas as pd

from p import process_pdf 


app = FastAPI()

origins = [
    "http://localhost:3000",  # Your Next.js Frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)
class ProviderInput(BaseModel):
    npi: str = Field(..., description="NPI ID to validate.")
class BatchInput(BaseModel):
    providers: List[ProviderInput]

class LicenseValidation(BaseModel):
    is_valid: bool = False
    status: str = "PENDING"  # PENDING, VALID, EXPIRED, NOT_FOUND
    expiry_date: Optional[str] = None
    last_verified: Optional[str] = None

class NpiDetailsOutput(BaseModel):
    npi_number: str
    status: str
    name: str
    address: str
    phone: str
    
    legal_name: Union[str, None] = None
    license_id: Union[str, None] = None
    specialty: Union[str, None] = None
    state: str = "N/A"
    validation_details: LicenseValidation = LicenseValidation()
    # --- NEW FIELDS ---
    email: Union[str, None] = None
    website_url: Union[str, None] = None
    # ------------------
    
    error_detail: Union[str, None] = None
# Structured Output (Merging OCR data with Validation results)
class StructuredDataOutput(BaseModel):
    # Data from OCR Agent
    ocr_extracted_name: str
    ocr_extracted_address: str
    ocr_extracted_phone: str
    ocr_extracted_email: Union[str, None] = None
    ocr_extracted_specialities: Union[str, None] = None
    
    # Core Data from NPI Registry (Validated Source)
    validated_npi: str
    validated_name: str
    validated_address: str
    validated_phone: str
    
    # --- ADDED: Comprehensive NPI Fields ---
    validated_legal_name: str
    validated_license_id: str
    validated_specialty: str
    # -------------------------------------

    # Validation Statuses and Scores (from concurrent agents)
    license_status: str
    google_maps_match: str
    website_scrape_status: str
    confidence_score: float
    
    # Metadata
    ocr_extraction_confidence: float


# --- Helper Functions (I/O & Mocks) ---

def save_upload_file_temp(upload_file: UploadFile, destination: str) -> str:
    """Saves the uploaded file to a temporary location for processing."""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

# 1. Data Validation Agent: NPI Lookup (Live API)
async def lookup_npi_registry(npi_number: str) -> dict:
    """Queries NPI Registry and prepares state-aware data for sequential validation."""
    NPI_API_URL = "https://npiregistry.cms.hhs.gov/api/"
    params = {"version": "2.1", "number": npi_number}
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(NPI_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result_count", 0) > 0:
                result = data["results"][0]
                basic = result.get("basic", {})
                
                # 1. Extract State Code from Primary Location (Crucial for Routing)
                practice_address_list = [addr for addr in result.get("addresses", []) 
                                         if addr.get("address_purpose") == "LOCATION"]
                address_info = practice_address_list[0] if practice_address_list else result.get("addresses", [{}])[0]
                state_code = address_info.get('state', 'N/A')
                
                # 2. Extract License and Specialty
                primary_taxonomy = [t for t in result.get("taxonomies", []) if t.get("primary") == True]
                taxonomy_info = primary_taxonomy[0] if primary_taxonomy else result.get("taxonomies", [{}])[0]

                # 3. Handle Names
                legal_name = basic.get('organization_name', f"{basic.get('first_name', '')} {basic.get('last_name', '')}").strip()
                
                # 4. Generate Mock Contact Data
                website_url = "https://www.example-provider.com"
                email_address = f"info@{legal_name.lower().replace(' ', '')}.com"

                # RETURN STRUCTURE (Aligned with upgraded Pydantic Model)
                return {
                    "status": "SUCCESS",
                    "npi_number": npi_number,
                    "name": legal_name,
                    "legal_name": legal_name,
                    "state": state_code, # Used for routing the next call
                    "license_id": taxonomy_info.get("license", "N/A"),
                    "specialty": taxonomy_info.get("desc", "N/A"),
                    "address": f"{address_info.get('address_1', '')}, {address_info.get('city', '')}, {state_code}",
                    "phone": address_info.get("telephone_number", "N/A"),
                    "email": email_address,
                    "website_url": website_url,
                    
                    # --- NEW: Placeholder for the next step in the pipeline ---
                    "validation_details": {
                        "is_valid": False,
                        "status": "PENDING",
                        "expiry_date": None,
                        "last_verified": None
                    }
                }
            else:
                return {"status": "NO_MATCH", "npi_number": npi_number, "error": "Zero results found."}
        except Exception as e:
            return {"status": "ERROR", "npi_number": npi_number, "error": str(e)}

async def scrape_contact_info(url: str) -> dict:
    """Mocks web scraping."""
    await asyncio.sleep(0.5) 
    if "mock-provider.com" in url:
        return {"status": "MATCH", "phone_scraped": "555-1212"}
    return {"status": "NO_CONTACT", "phone_scraped": "N/A"}

# 3. Data Validation Agent: License Lookup (MOCK)
async def mock_license_lookup(state_code: str, license_id: str) -> dict:
    """Mocks calling a state-specific license lookup API."""
    await asyncio.sleep(0.3) 
    if state_code == "NY" and license_id == "123456":
        return {"status": "VALID"}
    return {"status": "INVALID"}

# 4. Data Validation Agent: Google Maps Lookup (MOCK)
async def mock_google_maps_lookup(address: str) -> dict:
    """Mocks calling the Google Maps API."""
    await asyncio.sleep(0.4) 
    if "77 Water St" in address: 
         return {"status": "ADDRESS_MATCH"}
    return {"status": "FUZZY_MATCH"}


# --- Core Single-Provider Validation Logic (Used by all endpoints) ---

async def validate_single_provider(provider_data: ProviderInput) -> StructuredDataOutput:
    """
    Orchestrates NPI lookup and 3 concurrent validation calls.
    """
    
    # 1. Execute NPI Lookup (Sequential)
    npi_result = await lookup_npi_registry(provider_data.npi)

    if npi_result["status"] != "SUCCESS":
        raise HTTPException(
            status_code=404, 
            detail=f"NPI lookup failed for ID ({provider_data.npi}): {npi_result.get('error', npi_result['status'])}"
        )

    # 2. Extract fields for concurrent calls
    npi_address = npi_result["address"]
    npi_license = npi_result["license_id"]
    npi_state = npi_result["state"]
    npi_website = npi_result["website_url"] 

    # 3. SYNCHRONIZE 3 CONCURRENT CALLS
    scrape_task = scrape_contact_info(npi_website)
    license_task = mock_license_lookup(npi_state, npi_license)
    maps_task = mock_google_maps_lookup(npi_address) 
    
    scrape_result, license_result, maps_result = await asyncio.gather(
        scrape_task, license_task, maps_task
    )
    
    # 4. QA Agent Logic: Aggregate and Score
    confidence = 0.0
    if license_result["status"] == "VALID": confidence += 0.35
    if "MATCH" in scrape_result["status"]: confidence += 0.35
    if "MATCH" in maps_result["status"]: confidence += 0.30
    
    # Return structured output with NPI data and validation results
    return StructuredDataOutput(
        # OCR Extracted Fields (Defaulted for non-PDF calls)
        ocr_extracted_name="N/A (Non-PDF Input)",
        ocr_extracted_address="N/A (Non-PDF Input)",
        ocr_extracted_phone="N/A (Non-PDF Input)",
        ocr_extracted_email=None,
        ocr_extracted_specialities=None,

        # Validated/NPI Fields
        validated_npi=provider_data.npi,
        validated_name=npi_result["name"], 
        validated_address=npi_address,
        validated_phone=npi_result["phone"],
        
        # --- MAPPING NEW FIELDS ---
        validated_legal_name=npi_result["legal_name"],
        validated_license_id=npi_result["license_id"],
        validated_specialty=npi_result["specialty"],
        # ------------------------
        
        # Validation Statuses
        license_status=license_result["status"],
        google_maps_match=maps_result["status"],
        website_scrape_status=scrape_result["status"],
        confidence_score=round(confidence, 2),
        ocr_extraction_confidence=0.0
    )


# --- ENDPOINTS ---

# 1. PDF Pipeline Endpoint (The new primary flow)
# In main.py

# In main.py

# CHANGE: Response model is now a LIST
@app.post("/validate/pdf_pipeline", response_model=List[StructuredDataOutput])
async def run_single_pdf_pipeline(pdf_file: UploadFile = File(...)):
    """
    Multi-Provider Pipeline: 
    1. OCR 10-page PDF -> Extracts 10 Providers
    2. Runs Validation on ALL 10 concurrently
    """
    temp_file_path = f"temp_{pdf_file.filename}"
    temp_path = save_upload_file_temp(pdf_file, temp_file_path)
    
    # 1. Run OCR (This now returns a LIST of providers from all pages)
    ocr_generator = await asyncio.to_thread(process_pdf, temp_path)
    
    all_extracted_providers = []
    
    # Consume the generator to get all batches
    if hasattr(ocr_generator, "__iter__"):
        for batch in ocr_generator:
            all_extracted_providers.extend(batch)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not all_extracted_providers:
        raise HTTPException(status_code=400, detail="No NPIs found in any of the pages.")

    # 2. Prepare Validation Tasks for ALL providers found
    validation_tasks = []
    
    for ocr_data in all_extracted_providers:
        npi_id = ocr_data.get("npi_number") or ocr_data.get("NPI")
        
        if npi_id and npi_id != "N/A":
            # Pass the OCR data to the validator so we can merge it later
            task = validate_and_merge(npi_id, ocr_data)
            validation_tasks.append(task)

    # 3. Run all validations in parallel (Fast!)
    results = await asyncio.gather(*validation_tasks)
    
    return results

# --- HELPER FUNCTION TO KEEP CODE CLEAN ---
async def validate_and_merge(npi_id: str, ocr_data: dict) -> StructuredDataOutput:
    # Run the core check
    validation_output = await validate_single_provider(ProviderInput(npi=npi_id))
    
    # Merge OCR Metadata
    return validation_output.copy(update={
        "ocr_extracted_name": ocr_data.get("Provider Name (Legal name)", "N/A"),
        "ocr_extracted_address": ocr_data.get("Address", "N/A"),
        "ocr_extracted_phone": ocr_data.get("phone", "N/A"),
        "ocr_extraction_confidence": ocr_data.get("extraction_confidence", 0.0)
    })
# 2. Full Batch Processing Endpoint
@app.post("/validate/batch", response_model=List[StructuredDataOutput])
async def validate_provider_batch(batch_data: BatchInput):
    """
    Handles a batch of providers, running the entire validation pipeline 
    for all providers concurrently.
    """
    validation_tasks = []
    
    for provider in batch_data.providers:
        task = asyncio.create_task(
            validate_single_provider(provider) 
        )
        validation_tasks.append(task)
        
    try:
        results = await asyncio.gather(*validation_tasks)
    except HTTPException as e:
        print(f"Batch processing error: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e.detail}")

    return results

# 3. Simple JSON Validation Endpoint (for easy debugging without a file)
@app.post("/validate/single", response_model=StructuredDataOutput)
async def validate_single_json(provider_data: ProviderInput):
    """Validates a single provider using a JSON input (NPI ID)."""
    return await validate_single_provider(provider_data)

# --- 4. NPI-ONLY Batch Endpoint (Stress Test) ---
@app.post("/batch/npi_lookup", response_model=List[NpiDetailsOutput])
async def batch_npi_lookup(batch_data: BatchInput) -> List[NpiDetailsOutput]:
    """
    Runs the NPI lookup function for a batch of NPI IDs concurrently 
    and prepares state-specific routing data.
    """
    lookup_tasks = []
    
    for provider in batch_data.providers:
        task = asyncio.create_task(lookup_npi_registry(provider.npi))
        lookup_tasks.append(task)
        
    raw_results = await asyncio.gather(*lookup_tasks, return_exceptions=True)
    
    final_results = []
    
    for i, result in enumerate(raw_results):
        npi = batch_data.providers[i].npi
        
        # 1. Handle Critical Exceptions (Timeouts, Network failures)
        if isinstance(result, Exception):
             final_results.append(NpiDetailsOutput(
                npi_number=npi,
                status="CRITICAL_ERROR",
                name="N/A",
                address="N/A",
                phone="N/A",
                state="N/A", # Added state
                legal_name=None,
                license_id=None,
                specialty=None,
                email=None,
                website_url=None,
                error_detail=str(result),
                validation_details=LicenseValidation(status="ERROR") # Added nested object
            ))
             continue
        
        # 2. Format successful/NO_MATCH results
        # We use .get() to safely map the new 'state' and 'validation_details' fields
        final_results.append(NpiDetailsOutput(
            npi_number=npi,
            status=result["status"],
            name=result.get("name", "N/A"),
            address=result.get("address", "N/A"),
            phone=result.get("phone", "N/A"),
            state=result.get("state", "N/A"), # NEW: Crucial for State Routing
            
            legal_name=result.get("legal_name"), 
            license_id=result.get("license_id"), 
            specialty=result.get("specialty"), 
            email=result.get("email"),
            website_url=result.get("website_url"),
            
            error_detail=result.get("error"),
            
            # NEW: Mapping the nested validation block
            # This will show "PENDING" for now as we haven't run state lookups yet
            validation_details=result.get("validation_details", LicenseValidation())
        ))
        
    return final_results

# --- How to Run ---
# uvicorn main:app --reload