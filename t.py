from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(lang="en")

# Image path
image_path = "sample.jpg"

# Run OCR
results = ocr.ocr(image_path)
print(results)
# Print results
text = results[0]["rec_texts"]
print("Maybe hua")
print(text)