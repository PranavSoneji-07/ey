import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
import httpx 
import asyncio

import shutil
import pandas as pd

# Import the necessary synchronous function from your extraction file
from extraction_agent import process_pdf 

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Schemas (The Contract) ---

# Input data for all NPI lookups
class ProviderInput(BaseModel):
    npi: str = Field(..., description="NPI ID to validate.")

# Batch Input Schema for the /validate/batch endpoint
class BatchInput(BaseModel):
    providers: List[ProviderInput]

# --- NEW: Simplified NPI Output Schema (For /batch/npi_lookup) ---
# --- NEW: Simplified NPI Output Schema (MODIFIED) ---
# --- NpiDetailsOutput Schema in main.py (UPDATE THIS) ---
class NpiDetailsOutput(BaseModel):
    npi_number: str
    status: str
    name: str
    address: str
    phone: str
    
    legal_name: Union[str, None] = None
    license_id: Union[str, None] = None
    specialty: Union[str, None] = None
    
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
    """Queries NPI Registry for core demographics, including legal name, license, and specialty."""
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
                
                # Robust extraction of Address (prioritizing LOCATION)
                practice_address_list = [addr for addr in result.get("addresses", []) if addr.get("address_purpose") == "LOCATION"]
                address_info = practice_address_list[0] if practice_address_list else result.get("addresses", [{}])[0]
                
                # Extract primary taxonomy (license and specialty data)
                primary_taxonomy = [t for t in result.get("taxonomies", []) if t.get("primary") == True]
                taxonomy_info = primary_taxonomy[0] if primary_taxonomy else result.get("taxonomies", [{}])[0]

                # Extract the legal business name or full name
                legal_name = basic.get('organization_name', f"{basic.get('first_name', '')} {basic.get('last_name', '')}").strip()
                
                # Extract Website (if available)
                # The CMS API lists website in "endpoints" or "addresses" depending on version. 
                # We will use a mock for consistency in the demo.
                website_url = "https://www.example-provider.com" # MOCK URL for demo
                
                # MOCK Email as CMS API does not provide it
                email_address = f"info@{legal_name.lower().replace(' ', '')}.com" if legal_name else "default@provider.com"

                # Extract License ID and Specialty (now correctly pulled from taxonomy_info)
                license_id = taxonomy_info.get("license", "N/A")
                specialty = taxonomy_info.get("desc", taxonomy_info.get("code", "N/A"))

                return {
                    "status": "SUCCESS",
                    "name": legal_name, 
                    "address": f"{address_info.get('address_1', 'N/A')}, {address_info.get('city', '')}, {address_info.get('state', '')}",
                    "phone": address_info.get("telephone_number", "N/A"),
                    
                    # --- COMPREHENSIVE FIELDS (Updated) ---
                    "legal_name": legal_name,
                    "license_id": license_id,
                    "state_code": taxonomy_info.get("state", "N/A"),
                    "specialty": specialty, 
                    "website_url": website_url,
                    "email": email_address # NEW FIELD
                }
            else:
                return {"status": "NO_MATCH", "error": "API returned zero results for this ID."}
        except Exception as e:
            return {"status": "ERROR", "error": f"NPI Lookup Failed: {str(e)}"}
        
# 2. Information Enrichment Agent: Web Scraping (MOCK)
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
    npi_state = npi_result["state_code"]
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
@app.post("/validate/pdf_pipeline", response_model=StructuredDataOutput)
async def run_single_pdf_pipeline(pdf_file: UploadFile = File(...)):
    """
    Runs the full pipeline: PDF -> Extraction -> NPI Lookup -> Concurrent Validation.
    """
    temp_file_path = f"temp_{pdf_file.filename}"
    
    # --- 1. Data Extraction Agent (Sequential) ---
    temp_path = save_upload_file_temp(pdf_file, temp_file_path)
    
    # Use asyncio.to_thread to run the blocking/synchronous OCR function 
    ocr_data: Dict[str, Any] = await asyncio.to_thread(process_pdf, temp_path) 
    os.unlink(temp_path)
    
    # Extract the key field for validation
    npi_id = ocr_data.get("NPI")
    if not npi_id:
        raise HTTPException(status_code=400, detail="Extraction failed: Could not reliably find NPI ID in the document.")

    # 2. Orchestration Agent: Use the extracted NPI ID to run the validation flow
    provider_data = ProviderInput(npi=npi_id)
    
    # Call the core validation logic
    validation_output = await validate_single_provider(provider_data)
    
    # --- 3. Merge OCR Data with Validation Output ---
    
    # Use the validation output's data and override the dummy OCR fields
    return validation_output.copy(update={
        "ocr_extracted_name": ocr_data.get("Provider Name (Legal name)", "N/A"),
        "ocr_extracted_address": ocr_data.get("Address", "N/A"),
        "ocr_extracted_phone": ocr_data.get("Phone number", "N/A"),
        "ocr_extracted_email": ocr_data.get("Email address", None),
        "ocr_extracted_specialities": ocr_data.get("Specialities", None),
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
    Runs the NPI lookup function for a batch of NPI IDs concurrently.
    """
    lookup_tasks = []
    
    for provider in batch_data.providers:
        # Note: We still only pass NPI ID here for the pure NPI batch test
        task = asyncio.create_task(
            lookup_npi_registry(provider.npi) 
        )
        lookup_tasks.append(task)
        
    raw_results = await asyncio.gather(*lookup_tasks, return_exceptions=True)
    
    final_results = []
    
    for i, result in enumerate(raw_results):
        npi = batch_data.providers[i].npi
        
        # Handle exceptions (e.g., timeouts, network errors)
        if isinstance(result, Exception):
             final_results.append(NpiDetailsOutput(
                npi_number=npi,
                status="CRITICAL_ERROR",
                name="N/A",
                address="N/A",
                phone="N/A",
                legal_name=None, # Changed from "N/A" to None to match new schema type
                license_id=None,
                specialty=None,
                email=None,          # NEW: Map None on error
                website_url=None,    # NEW: Map None on error
                error_detail=str(result)
            ))
             continue
        
        # Format successful/NO_MATCH results (Updated mapping)
        final_results.append(NpiDetailsOutput(
            npi_number=npi,
            status=result["status"],
            name=result.get("name", "N/A"),
            address=result.get("address", "N/A"),
            phone=result.get("phone", "N/A"),
            
            # --- Mapped from lookup_npi_registry result ---
            legal_name=result.get("legal_name", None), 
            license_id=result.get("license_id", None), 
            specialty=result.get("specialty", None), 
            email=result.get("email", None),          # NEW: Map email from result
            website_url=result.get("website_url", None), # NEW: Map website_url from result
            # ----------------------------------------------
            
            error_detail=result.get("error", None)
        ))
        
    return final_results

# --- How to Run ---
# uvicorn main:app --reload