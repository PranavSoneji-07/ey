import os
import io
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import spacy
"""
--- OPTIONAL: Google Vision imports (if you use fallback) ---
try:
    from google.cloud import vision
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False
"""


# OCR
PADDLE_LANG = "en"
PADDLE_OCR = PaddleOCR(lang=PADDLE_LANG)
OCR_CONFIDENCE_THRESHOLD = 0.75  # if below this, try Google Vision

# Regex patterns
NPI_REGEX = re.compile(r"(?:NPI[:\s#-]*)?(\d{10})")
PHONE_REGEX = re.compile(r"(?<!\d)([2-9]\d{9})(?!\d)")
LICENSE_REGEX = re.compile(
    r"(?:License(?:\s*No\.?| #| Number)?[:\s-]*)([A-Za-z0-9-]+)",
    re.IGNORECASE,
)

EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

# spaCy model
NLP = None

def get_nlp():
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load("en_core_web_trf")
        except OSError as e:
            # Optional: fallback if trf model not installed
            print("[WARN] Could not load en_core_web_trf:", e)
          #  print("[WARN] Falling back to en_core_web_sm")
         #   NLP = spacy.load("en_core_web_sm")
    return NLP


# ==========================
# PDF -> IMAGES
# ==========================

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Any]:
    """
    Convert PDF to a list of PIL Images.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    return images


# ==========================
# OCR HELPERS
# ==========================

import numpy as np   # put this at the top

def run_paddle_ocr(image):
    img_np = np.array(image)
    result = PADDLE_OCR.predict(img_np)

    if not result:
        return [], "paddle"

    texts = result[0]["rec_texts"]
    scores = result[0]["rec_scores"]
    print(texts)
    text_blocks = []
    for txt, score in zip(texts, scores):
        if not txt.strip():
            continue
        text_blocks.append({
            "text": txt.strip(),  
            "conf": float(score),   
        })


    return text_blocks, "paddle"



"""
 def run_google_vision(image) -> Tuple[List[Dict[str, Any]], float]:
    
    Run Google Vision OCR on a PIL image.
    You must have GOOGLE_APPLICATION_CREDENTIALS set and google-cloud-vision installed.
    Returns: (list of text blocks, avg_confidence)
    
    if not HAS_GOOGLE_VISION:
        # Fallback stub if library not installed
        return [], 0.0

    client = vision.ImageAnnotatorClient()

    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    content = img_byte_arr.getvalue()

    image_obj = vision.Image(content=content)
    response = client.document_text_detection(image=image_obj)
    annotation = response.full_text_annotation

    text_blocks = []
    confidences = []

    for page in annotation.pages:
        for block in page.blocks:
            block_text = []
            block_confidences = []
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join(
                        [symbol.text for symbol in word.symbols]
                    )
                    block_text.append(word_text)
                    if word.confidence is not None:
                        block_confidences.append(float(word.confidence))

            txt = " ".join(block_text).strip()
            if not txt:
                continue

            # Approximate bounding box
            box = [
                (v.x, v.y) for v in block.bounding_box.vertices
            ]

            if block_confidences:
                c = sum(block_confidences) / len(block_confidences)
                confidences.append(c)
            else:
                c = 0.0

            text_blocks.append({
                "text": txt,
                "conf": c,
                "box": box,
            })

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text_blocks, avg_conf
"""


def run_ocr_with_fallback(image, threshold: float = OCR_CONFIDENCE_THRESHOLD):
    """
    Run PaddleOCR, and if quality is poor, fallback to Google Vision.
    Returns: (text_blocks, avg_conf, engine_used)
    """
    text_blocks, engine_used = run_paddle_ocr(image)
    # avg_conf = paddle_conf

    # if paddle_conf < threshold and HAS_GOOGLE_VISION:
    #     gv_blocks, gv_conf = run_google_vision(image)
    #     if gv_conf > paddle_conf:
    #         text_blocks = gv_blocks
    #         avg_conf = gv_conf
    #         engine_used = "google_vision"

    return text_blocks, engine_used

"""
def assemble_text(text_blocks: List[Dict[str, Any]]) -> str:
    
    Assemble text blocks into a single text string.
    Sort by Y then X for a more natural reading order.
    
    def _key(block):
        # box is list of 4 points; take top-left
        box = block["box"]
        if not box:
            return (0, 0)
        x = box[0][0]
        y = box[0][1]
        return (y, x)

    sorted_blocks = sorted(text_blocks, key=_key)
    return "\n".join(b["text"] for b in sorted_blocks if b["text"].strip())
"""


# NLP / EXTRACTION


def extract_provider_name(doc: spacy.tokens.Doc) -> Tuple[str, float]:
    """
    Heuristic:
    - Prefer ORG entities that contain 'Hospital', 'Clinic', 'Medical', 'Center', etc.
    - Otherwise take the longest ORG entity.
    """
    keywords = ["hospital", "clinic", "medical", "centre", "center", "health", "provider"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]

    best = None
    best_score = 0.0

    for ent in orgs:
        text_lower = ent.text.lower()
        score = len(ent.text)  # base: length
        if any(k in text_lower for k in keywords):
            score += 20  # strong bonus if keyword present
        if score > best_score:
            best_score = score
            best = ent

    if best is None and orgs:
        best = max(orgs, key=lambda e: len(e.text))
        best_score = len(best.text)

    if best:
        # normalize confidence to 0.5–0.95 range
        conf = min(0.95, 0.5 + best_score / 50.0)
        return best.text.strip(), conf

    return None, 0.0


def extract_address(text: str, provider_name: str = None) -> Tuple[str, float]:
    """
    Very simple heuristic address extractor:
    - Look for lines with address-ish clues (street, road, city, zip, etc.)
    - If provider_name is present, take lines after it.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate_lines = []

    # rough indicators
    addr_keywords = [
        "street", "st.", "st,", "road", "rd", "ave", "avenue",
        "blvd", "block", "city", "state", "zip", "pincode",
        "pin code", "suite", "floor", "flr", "building"
    ]

    # We’ll also try to limit to block around provider name
    start_idx = 0
    if provider_name:
        for i, ln in enumerate(lines):
            if provider_name.lower() in ln.lower():
                # start looking from this line onwards
                start_idx = i
                break

    for ln in lines[start_idx:]:
        lower = ln.lower()
        if any(k in lower for k in addr_keywords):
            candidate_lines.append(ln)

    if not candidate_lines:
        return None, 0.0

    address = ", ".join(candidate_lines)
    return address, 0.7


def extract_specialities(text: str) -> Tuple[str, float]:
    """
    Extract specialities:
    - Look for lines containing 'Speciality', 'Specialties', 'Specialization', etc.
    - Return comma-separated list.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keywords = ["speciality", "specialties", "speciality:", "specializations",
                "specialization", "specialty", "department", "dept"]
    speciality_lines = []

    for ln in lines:
        lower = ln.lower()
        if any(k in lower for k in keywords):
            # strip label part
            parts = re.split(r":", ln, maxsplit=1)
            if len(parts) == 2:
                val = parts[1].strip()
            else:
                val = ln
            speciality_lines.append(val)

    if not speciality_lines:
        return None, 0.0

    # Merge and normalize
    full = ", ".join(speciality_lines)
    # Split on comma/semicolon for cleaning
    items = [x.strip() for x in re.split(r"[;,]", full) if x.strip()]
    items = sorted(set(items), key=lambda s: s.lower())
    result = ", ".join(items)
    return result, 0.85


def extract_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """
    Main extractor: NPI, Provider Name (Legal name), Address, Phone number,
    License number, Specialities.
    """
    nlp = get_nlp()
    doc = nlp(text)
        # Pre-clean OCR text for emails
    clean_text = re.sub(r"\s*@\s*", "@", text)
    text = re.sub(r"\s*\.\s*", ".", clean_text)

    data = {
        "NPI": None,
        "Provider Name (Legal name)": None,
        "Address": None,
        "Phone number": None,
        "Email address": None,
        "License number": None,
        "Specialities": None,
    }

    conf = {k: 0.0 for k in data.keys()}

    # Provider name
    provider_name, pn_conf = extract_provider_name(doc)
    data["Provider Name (Legal name)"] = provider_name
    conf["Provider Name (Legal name)"] = pn_conf

    # NPI
    npi_match = NPI_REGEX.search(text)
    if npi_match:
        data["NPI"] = npi_match.group(1)
        conf["NPI"] = 0.95

    # Phone number
    phone_match = PHONE_REGEX.search(text)
    if phone_match:
        data["Phone number"] = phone_match.group(1).strip()
        conf["Phone number"] = 0.9

    # License number
    lic_match = LICENSE_REGEX.search(text)
    if lic_match:
        data["License number"] = lic_match.group(1).strip()
        conf["License number"] = 0.9
        
    # Email address
    email_match = EMAIL_REGEX.search(text)
    if email_match:
        email = normalize_email(email_match.group(0))
        data["Email address"] = email
        conf["Email address"] = 0.95

    # Specialities
    specs, specs_conf = extract_specialities(text)
    if specs:
        data["Specialities"] = specs
        conf["Specialities"] = specs_conf

    # Address (use provider name for context)
    address, addr_conf = extract_address(text, provider_name)
    if address:
        data["Address"] = address
        conf["Address"] = addr_conf

    # overall confidence
    overall_conf = sum(conf.values()) / len(conf) if conf else 0.0

    return data, conf, overall_conf


# ==========================
# NORMALIZATION (pandas)
# ==========================

def normalize_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def normalize_npi(npi: Any) -> Any:
    if not isinstance(npi, str):
        return npi
    digits = "".join(ch for ch in npi if ch.isdigit())
    if len(digits) == 10:
        return digits
    return digits or None


def normalize_phone(phone: Any) -> Any:
    if not isinstance(phone, str):
        return phone
    digits = "".join(ch for ch in phone if ch.isdigit())
    if not digits:
        return None
    # very simple formatter, adjust to your region
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    return digits  # leave as-is if not 10 digits


def normalize_email(raw: str) -> str:
    """
    Fix common OCR email issues:
    - spaces around @ and .
    - ' dot ' → '.'
    """
    if not raw:
        return None

    email = raw.lower()
    email = re.sub(r"\s*@\s*", "@", email)
    email = re.sub(r"\s*\.\s*", ".", email)
    email = email.replace(" dot ", ".")
    return email


def normalize_specialities(specs: Any) -> Any:
    if not isinstance(specs, str):
        return specs
    items = [x.strip() for x in re.split(r"[;,]", specs) if x.strip()]
    # title-case for readability
    items = [item.title() for item in items]
    # remove duplicates
    items = sorted(set(items), key=lambda s: s.lower())
    return ", ".join(items)


def normalize_record(raw_data: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    """
    raw_data: fields extracted by NLP
    meta: {engine, ocr_conf, extraction_conf, source_file}
    Returns a single-row DataFrame.
    """
    df = pd.DataFrame([raw_data])

    # Strip/clean all text columns
    for col in df.columns:
        df[col] = df[col].apply(normalize_text)

    # Field-specific normalization
    if "NPI" in df.columns:
        df["NPI"] = df["NPI"].apply(normalize_npi)

    if "Phone number" in df.columns:
        df["Phone number"] = df["Phone number"].apply(normalize_phone)

    if "Specialities" in df.columns:
        df["Specialities"] = df["Specialities"].apply(normalize_specialities)

    # Attach meta
    df["ocr_engine_used"] = meta.get("engine")
    df["ocr_quality_score"] = meta.get("ocr_conf")
    df["extraction_confidence"] = meta.get("extraction_conf")
    df["source_file"] = meta.get("source_file")

    return df


# ==========================
# MAIN PIPELINE
# ==========================

def process_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Full pipeline:
    - PDF -> images
    - OCR (Paddle + Google Vision fallback)
    - Text assembly (merge all pages)
    - NLP extraction
    - Normalization
    Returns a single-row DataFrame for the first provider in the document.
    """
    images = pdf_to_images(pdf_path)

    all_page_texts = []
    engine_used = None
    ocr_scores = []

    for img in images:
        blocks, engine = run_ocr_with_fallback(img)
        page_text = "\n".join(b["text"] for b in blocks)
        all_page_texts.append(page_text)
       # ocr_scores.append(score)
        # If multiple pages, you could store per-page engines;
        # for simplicity just track the last one that was used.
        engine_used = engine

    full_text = "\n".join(all_page_texts)

    raw_data, field_conf, extraction_conf = extract_fields(full_text)

    meta = {
        "engine": engine_used,
        #"ocr_conf": sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0.0,
        "extraction_conf": extraction_conf,
        "source_file": os.path.abspath(pdf_path),
    }

    df = normalize_record(raw_data, meta)
    return df


# ==========================
# CLI / DEMO
# ==========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract provider details from a scanned PDF."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument(
        "--out",
        help="Output CSV file path",
        default="provider_output.csv",
    )
    args = parser.parse_args()

    df_result = process_pdf(args.pdf_path)
    print("Extraction Result:")
    print(df_result.to_string(index=False))

    df_result.to_csv(args.out, index=False)
    print(f"\nSaved normalized data to: {args.out}")
