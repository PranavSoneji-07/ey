import os
import io
import re
import numpy as np
import pandas as pd
import spacy
from typing import List, Dict, Any, Tuple
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================
# CONFIGURATION & REGEX
# ==========================
OCR_CONFIDENCE_THRESHOLD = 0.75

NPI_REGEX = re.compile(r"(?:NPI[:\s#-]*)?(\d{10})")
PHONE_REGEX = re.compile(r"(?<!\d)([2-9]\d{9})(?!\d)")
LICENSE_REGEX = re.compile(
    r"(?:License(?:\s*No\.?| #| Number)?[:\s-]*)([A-Za-z0-9-]+)",
    re.IGNORECASE,
)
EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

# Global NLP variable for the main process
_NLP_MODEL = None

def get_nlp():
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            _NLP_MODEL = spacy.load("en_core_web_trf")
        except OSError:
            # Fallback if transformer model isn't installed
            _NLP_MODEL = spacy.load("en_core_web_sm")
    return _NLP_MODEL

# ==========================
# WORKER FUNCTIONS (Run on separate cores)
# ==========================

def worker_ocr_task(image_array):
    """
    This function runs on a single CPU core. 
    It initializes its own PaddleOCR instance to avoid Windows serialization issues.
    """
    # Initialize PaddleOCR locally within the process
    # cls=False speeds up processing if you don't need to detect text orientation
    ocr_engine = PaddleOCR(lang='en')
    
    # result is a list: [ [ [box, (text, score)], ... ] ]
    result = ocr_engine.ocr(image_array)
    
    page_blocks = []
    if result and result[0]:
        for line in result[0]:
            text_str = line[1][0]
            confidence = line[1][1]
            if text_str.strip():
                page_blocks.append({
                    "text": text_str.strip(),
                    "conf": float(confidence)
                })
    
    # Return the assembled text for this specific page
    return "\n".join(b["text"] for b in page_blocks)

# ==========================
# EXTRACTION HEURISTICS
# ==========================

def extract_provider_name(doc) -> Tuple[str, float]:
    keywords = ["hospital", "clinic", "medical", "centre", "center", "health", "provider"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    best = None
    best_score = 0
    for ent in orgs:
        text_lower = ent.text.lower()
        score = len(ent.text)
        if any(k in text_lower for k in keywords):
            score += 20
        if score > best_score:
            best_score = score
            best = ent
    if best:
        return best.text.strip(), min(0.95, 0.5 + best_score / 50.0)
    return None, 0.0

def extract_address(text: str, provider_name: str = None) -> Tuple[str, float]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    addr_keywords = ["street", "st.", "road", "rd", "ave", "avenue", "blvd", "city", "state", "zip"]
    candidate_lines = []
    start_idx = 0
    if provider_name:
        for i, ln in enumerate(lines):
            if provider_name.lower() in ln.lower():
                start_idx = i
                break
    for ln in lines[start_idx:]:
        if any(k in ln.lower() for k in addr_keywords):
            candidate_lines.append(ln)
    return (", ".join(candidate_lines), 0.7) if candidate_lines else (None, 0.0)

def extract_specialities(text: str) -> Tuple[str, float]:
    keywords = ["speciality", "specialties", "specialization", "specialty", "department"]
    speciality_lines = []
    for ln in text.splitlines():
        if any(k in ln.lower() for k in keywords):
            parts = re.split(r":", ln, maxsplit=1)
            val = parts[1].strip() if len(parts) == 2 else ln
            speciality_lines.append(val)
    if not speciality_lines: return None, 0.0
    items = sorted(set([x.strip().title() for x in re.split(r"[;,]", ", ".join(speciality_lines)) if x.strip()]))
    return ", ".join(items), 0.85

def extract_fields(text: str) -> Dict[str, Any]:
    nlp = get_nlp()
    doc = nlp(text)
    
    # Pre-clean for common OCR email artifacts
    clean_text = re.sub(r"\s*@\s*", "@", text)
    clean_text = re.sub(r"\s*\.\s*", ".", clean_text)

    p_name, _ = extract_provider_name(doc)
    
    data = {
        "NPI": (NPI_REGEX.search(text) or [None, None])[1],
        "Provider Name (Legal name)": p_name,
        "Address": extract_address(text, p_name)[0],
        "Phone number": (PHONE_REGEX.search(text) or [None])[0],
        "Email address": (EMAIL_REGEX.search(clean_text) or [None])[0],
        "License number": (LICENSE_REGEX.search(text) or [None, None])[1],
        "Specialities": extract_specialities(text)[0],
    }
    return data

# ==========================
# NORMALIZATION
# ==========================

def normalize_record(raw_data: Dict[str, Any], pdf_path: str) -> pd.DataFrame:
    df = pd.DataFrame([raw_data])
    
    # Clean whitespace
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Phone Formatting
    if df["Phone number"].iloc[0]:
        d = "".join(filter(str.isdigit, df["Phone number"].iloc[0]))
        if len(d) == 10:
            df["Phone number"] = f"({d[:3]}) {d[3:6]}-{d[6:]}"
            
    df["source_file"] = os.path.abspath(pdf_path)
    return df

# ==========================
# MAIN PIPELINE
# ==========================

def process_pdf(pdf_path: str):
    print(f"[*] Converting PDF to images: {os.path.basename(pdf_path)}")
    images = convert_from_path(pdf_path, dpi=300)
    
    # Convert PIL images to numpy arrays once to pass to workers efficiently
    image_arrays = [np.array(img) for img in images]
    
    all_page_texts = [None] * len(images)
    num_cores = os.cpu_count()
    
    print(f"[*] Launching parallel OCR on {len(images)} pages using {num_cores} cores...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map each image array to the worker task
        future_to_page = {executor.submit(worker_ocr_task, img): i for i, img in enumerate(image_arrays)}
        
        for future in as_completed(future_to_page):
            page_idx = future_to_page[future]
            try:
                page_text = future.result()
                all_page_texts[page_idx] = page_text
                print(f"    [+] Page {page_idx + 1} complete")
            except Exception as e:
                print(f"    [!] Page {page_idx + 1} failed: {e}")
                all_page_texts[page_idx] = ""

    full_document_text = "\n".join(filter(None, all_page_texts))
    
    print("[*] Running NLP Extraction...")
    raw_data = extract_fields(full_document_text)
    
    return normalize_record(raw_data, pdf_path)

# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    # Crucial for Windows multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Multi-core Provider Data Extractor")
    parser.add_argument("pdf_path", help="Path to scanned PDF")
    parser.add_argument("--out", default="extracted_provider.csv", help="Output filename")
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found.")
    else:
        result_df = process_pdf(args.pdf_path)
        
        print("\n" + "="*30)
        print("EXTRACTION COMPLETE")
        print("="*30)
        print(result_df.transpose()) # Transpose for better CLI visibility
        
        result_df.to_csv(args.out, index=False)
        print(f"\n[!] Data saved to {args.out}")