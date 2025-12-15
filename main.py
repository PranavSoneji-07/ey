from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
import httpx 
import asyncio
import os
import shutil
import pandas as pd

# Import the necessary synchronous function from your extraction file
# Ensure this file exists and contains the process_pdf function.
from extraction_agent import process_pdf 

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Schemas (The Contract) ---

# Input data for the batch endpoint (only NPI ID needed for validation flow)
class ProviderInput(BaseModel):
    npi: str = Field(..., description="NPI ID to validate.")

# Batch Input Schema for the /validate/batch endpoint
class BatchInput(BaseModel):
    providers: List[ProviderInput]

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
    # Note: Using shutil.copyfileobj with UploadFile.file (a SpooledTemporaryFile) 
    # is the standard blocking way to save the file.
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

# 1. Data Validation Agent: NPI Lookup (Live API)
async def lookup_npi_registry(npi_number: str) -> dict:
    """Queries NPI Registry for core demographics."""
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
                
                # Robust extraction based on NPI documentation structure
                practice_address_list = [addr for addr in result.get("addresses", []) if addr.get("address_purpose") == "LOCATION"]
                address_info = practice_address_list[0] if practice_address_list else result.get("addresses", [{}])[0]
                license_info = result.get("taxonomies", [{}])[0]
                
                return {
                    "status": "SUCCESS",
                    "name": basic.get('organization_name', f"{basic.get('first_name', '')} {basic.get('last_name', '')}").strip(),
                    "address": f"{address_info.get('address_1', 'N/A')}, {address_info.get('city', '')}, {address_info.get('state', '')}",
                    "phone": address_info.get("telephone_number", "N/A"),
                    "license_id": license_info.get("license", "N/A"),
                    "state_code": license_info.get("state", "N/A"),
                    "website_url": "http://www.mock-provider.com" # Placeholder for subsequent agents
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


# --- Core Single-Provider Validation Logic (Used by both endpoints) ---

async def validate_single_provider(provider_data: ProviderInput) -> StructuredDataOutput:
    """
    Orchestrates NPI lookup and 3 concurrent validation calls.
    """
    
    # 1. Execute NPI Lookup (Sequential)
    npi_result = await lookup_npi_registry(provider_data.npi)

    if npi_result["status"] != "SUCCESS":
        # Raise an exception if the NPI lookup fails (handles failures inside batch)
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
    
    # 4. QA Agent Logic: Aggregate and Score (using dummy OCR data for full output)
    confidence = 0.0
    if license_result["status"] == "VALID": confidence += 0.35
    if "MATCH" in scrape_result["status"]: confidence += 0.35
    if "MATCH" in maps_result["status"]: confidence += 0.30
    
    # Return structured output with NPI data and validation results
    return StructuredDataOutput(
        # OCR Extracted Fields (MOCKED for /validate/single)
        ocr_extracted_name="N/A (Single JSON Input)",
        ocr_extracted_address="N/A (Single JSON Input)",
        ocr_extracted_phone="N/A (Single JSON Input)",
        ocr_extracted_email=None,
        ocr_extracted_specialities=None,

        # Validated/NPI Fields
        validated_npi=provider_data.npi,
        validated_name=npi_result["name"], 
        validated_address=npi_address,
        validated_phone=npi_result["phone"],
        
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
    
    # Use asyncio.to_thread to run the blocking/synchronous OCR function without
    # freezing the main FastAPI event loop.
    
    # Clean up the temporary file
    
    
    # Convert DataFrame to dictionary for easy access
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

# 2. Batch Processing Endpoint (for high-performance scaling demo)
@app.post("/validate/batch", response_model=List[StructuredDataOutput])
async def validate_provider_batch(batch_data: BatchInput):
    """
    Handles a batch of providers, running the entire validation pipeline 
    for all providers concurrently.
    """
    
    validation_tasks = []
    
    for provider in batch_data.providers:
        # Create an asynchronous task for EVERY provider
        task = asyncio.create_task(
            validate_single_provider(provider) 
        )
        validation_tasks.append(task)
        
    # Run ALL provider validation pipelines concurrently
    try:
        results = await asyncio.gather(*validation_tasks)
    except HTTPException as e:
        # Graceful error handling for the batch
        print(f"Batch processing error: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e.detail}")

    return results

# 3. Simple JSON Validation Endpoint (for easy debugging without a file)
@app.post("/validate/single", response_model=StructuredDataOutput)
async def validate_single_json(provider_data: ProviderInput):
    """Validates a single provider using a JSON input (NPI ID)."""
    return await validate_single_provider(provider_data)

# --- How to Run ---
# uvicorn main:app --reload