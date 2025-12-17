import os
import io
import re
from typing import List, Dict, Any, Tuple, Generator
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
import spacy

# OCR Configuration
PADDLE_LANG = "en"
PADDLE_OCR = PaddleOCR(lang=PADDLE_LANG)
OCR_CONFIDENCE_THRESHOLD = 0.75

# Regex patterns
NPI_REGEX = re.compile(r"(?:npi|number|numbex)?[:\s#-]*([12]\d{9})", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?<!\d)([2-9]\d{9})(?!\d)")
LICENSE_REGEX = re.compile(
    r"(?:license|lic|id)(?:\s*id|\s*no\.?|\s*number)?[:\s#-]+(?=.*\d)([A-Za-z0-9-]{4,15})", 
    re.IGNORECASE
)
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Words that OCR frequently misidentifies as the actual ID
INVALID_IDS = {"license", "number", "npi", "null", "none", "id", "provider", "numbex"}

# spaCy model loader
NLP = None
def get_nlp():
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load("en_core_web_trf")
        except OSError:
            print("[WARN] Could not load en_core_web_trf.")
    return NLP

# ==========================
# PDF & OCR HELPERS
# ==========================

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Any]:
    return convert_from_path(pdf_path, dpi=dpi)

def run_paddle_ocr(image):
    img_np = np.array(image)
    result = PADDLE_OCR.predict(img_np)
    if not result: return [], "paddle"
    
    texts = result[0]["rec_texts"]
    scores = result[0]["rec_scores"]
    text_blocks = [{"text": txt.strip(), "conf": float(score)} 
                   for txt, score in zip(texts, scores) if txt.strip()]
    return text_blocks, "paddle"

def run_ocr_with_fallback(image, threshold: float = OCR_CONFIDENCE_THRESHOLD):
    return run_paddle_ocr(image)

# ==========================
# NORMALIZATION FUNCTIONS
# ==========================

def normalize_text(value: Any) -> Any:
    if not isinstance(value, str): return value
    return re.sub(r"\s+", " ", value).strip()

def normalize_npi(npi: Any) -> Any:
    if not isinstance(npi, str): return npi
    digits = "".join(ch for ch in npi if ch.isdigit())
    return digits if len(digits) == 10 else None

def normalize_phone(phone: Any) -> Any:
    if not isinstance(phone, str): return phone
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) == 10:
        # UPDATED: XXX-XXX-XXXX format
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:]}"
    return digits if digits else None

def normalize_email(raw: str) -> str:
    if not raw: return None
    email = raw.lower()
    email = re.sub(r"\s*@\s*", "@", email)
    email = re.sub(r"\s*\.\s*", ".", email)
    return email.replace(" dot ", ".")

def normalize_specialities(specs: Any) -> Any:
    if not isinstance(specs, str): return specs
    items = [x.strip().title() for x in re.split(r"[;,]", specs) if x.strip()]
    return ", ".join(sorted(set(items), key=lambda s: s.lower()))

# ==========================
# EXTRACTION LOGIC
# ==========================

def extract_provider_name(doc: spacy.tokens.Doc) -> Tuple[str, float]:
    keywords = ["hospital", "clinic", "medical", "centre", "center", "health", "provider"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    best = None
    best_score = 0.0
    for ent in orgs:
        score = len(ent.text) + (20 if any(k in ent.text.lower() for k in keywords) else 0)
        if score > best_score:
            best_score, best = score, ent
    if not best and orgs:
        best = max(orgs, key=lambda e: len(e.text))
        best_score = len(best.text)
    return (best.text.strip(), min(0.95, 0.5 + best_score / 50.0)) if best else (None, 0.0)

def extract_address(text: str, provider_name: str = None) -> Tuple[str, float]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    addr_keywords = ["street", "st.", "road", "ave", "blvd", "city", "state", "zip", "suite"]
    start_idx = 0
    if provider_name:
        for i, ln in enumerate(lines):
            if provider_name.lower() in ln.lower():
                start_idx = i
                break
    candidates = [ln for ln in lines[start_idx:] if any(k in ln.lower() for k in addr_keywords)]
    return (", ".join(candidates), 0.7) if candidates else (None, 0.0)

def extract_specialities(text: str) -> Tuple[str, float]:
    keywords = ["speciality", "specialties", "specialization", "specialty", "dept"]
    found = []
    for ln in text.splitlines():
        if any(k in ln.lower() for k in keywords):
            parts = re.split(r":", ln, maxsplit=1)
            found.append(parts[1].strip() if len(parts) == 2 else ln.strip())
    return (", ".join(found), 0.85) if found else (None, 0.0)

def extract_fields(text: str) -> Tuple[Dict[str, Any], float]:
    clean_text = " ".join(text.split())
    text_for_ids = re.sub(r'(\d)\s+(\d)', r'\1\2', clean_text)
    
    # NEW: Updated Column Names
    data = {
        "npi_number": None, 
        "Provider Name (Legal name)": None, 
        "Address": None,
        "phone": None, 
        "Email address": None, 
        "license_id": None, 
        "Specialities": None
    }
    
    # NPI
    npi_m = NPI_REGEX.search(text_for_ids)
    if npi_m: data["npi_number"] = npi_m.group(1)
    
    # License ID
    lic_m = LICENSE_REGEX.search(clean_text)
    if lic_m:
        val = lic_m.group(1).strip()
        if val.lower() not in INVALID_IDS and val != data["npi_number"]:
            data["license_id"] = val
            
    # Phone
    ph_m = PHONE_REGEX.search(text_for_ids)
    if ph_m: data["phone"] = ph_m.group(1)
    
    # Email
    em_m = EMAIL_REGEX.search(clean_text)
    if em_m: data["Email address"] = normalize_email(em_m.group(0))
    
    # NLP
    doc = get_nlp()(re.sub(r'\s+', ' ', text).strip())
    data["Provider Name (Legal name)"], _ = extract_provider_name(doc)
    data["Address"], _ = extract_address(text, data["Provider Name (Legal name)"])
    data["Specialities"], _ = extract_specialities(text)
    
    return data, 0.8

# ==========================
# MAIN PIPELINE
# ==========================

def process_pdf(pdf_path: str, batch_size: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
    images = pdf_to_images(pdf_path)
    current_batch = []
    source_abs = os.path.abspath(pdf_path)
    
    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        blocks, engine = run_ocr_with_fallback(img)
        page_text = " ".join(b["text"] for b in blocks)
        
        if not page_text.strip(): continue

        raw_data, ext_conf = extract_fields(page_text)

        # Apply Normalization before batching
        raw_data["npi_number"] = normalize_npi(raw_data["npi_number"])
        raw_data["phone"] = normalize_phone(raw_data["phone"])
        raw_data["Specialities"] = normalize_specialities(raw_data["Specialities"])
        for k in ["Provider Name (Legal name)", "Address", "license_id"]:
            raw_data[k] = normalize_text(raw_data[k])

        meta = {
            "ocr_engine_used": engine,
            "extraction_confidence": ext_conf,
            "source_file": source_abs,
            "page_number": i + 1 
        }

        record = {**raw_data, **meta}
        record = {k: (v if v is not None else "N/A") for k, v in record.items()}
        current_batch.append(record)

        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []

    if current_batch: yield current_batch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--out", default="provider_output.csv")
    args = parser.parse_args()

    all_data = []
    for batch in process_pdf(args.pdf_path, batch_size=10):
        for provider in batch:
            print(f"Found: {provider.get('Provider Name (Legal name)')} | NPI: {provider.get('npi_number')}")
            all_data.append(provider)

    if all_data:
        pd.DataFrame(all_data).to_csv(args.out, index=False)
        print(f"\nSUCCESS: Saved to {args.out}")