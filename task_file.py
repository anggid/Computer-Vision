"""
PDF Construction Image Object Detection
---------------------------------------
- Extracts only images (ignores text) from PDF
- Runs Hugging Face DETR object detection
- Filters predictions >= 80% confidence
- Outputs:
    1. Structured JSON results
    2. Annotated image per page
"""

import fitz  # PyMuPDF
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


# ==========================================================
# CONFIGURATION
# ==========================================================
PDF_PATH = "sample.pdf"
MODEL_NAME = "facebook/detr-resnet-50"  # Better spatial detection
CONFIDENCE_THRESHOLD = 0.80
OUTPUT_JSON = "construction_detection_results.json"


# ==========================================================
# LOAD MODEL (Spatial-aware DETR)
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ==========================================================
# PDF IMAGE EXTRACTION (IGNORE TEXT)
# ==========================================================
def extract_images_from_pdf(pdf_path):
    """
    Extract embedded images from each page.
    Text content is ignored.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        images = []

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            images.append(image)

        pages.append(images)

    return pages


# ==========================================================
# OBJECT DETECTION
# ==========================================================
def detect_objects(image):
    """
    Run object detection using DETR.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=CONFIDENCE_THRESHOLD
    )[0]

    return results


# ==========================================================
# DRAW BOUNDING BOXES
# ==========================================================
def annotate_image(image, detections):
    """
    Draw bounding boxes and labels on image.
    """
    draw = ImageDraw.Draw(image)

    for score, label, box in zip(
        detections["scores"],
        detections["labels"],
        detections["boxes"]
    ):
        xmin, ymin, xmax, ymax = box.tolist()
        class_name = model.config.id2label[label.item()]
        confidence = float(score)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text(
            (xmin, ymin - 10),
            f"{class_name} ({confidence:.2f})",
            fill="red"
        )

    return image


# ==========================================================
# MAIN PROCESS
# ==========================================================
def process_pdf(pdf_path):
    pages = extract_images_from_pdf(pdf_path)
    structured_results = []
    object_counter = 1

    for page_index, images in enumerate(pages):
        print(f"Processing Page {page_index + 1}...")

        for img_index, image in enumerate(images):
            detections = detect_objects(image)

            # Save structured data
            for score, label, box in zip(
                detections["scores"],
                detections["labels"],
                detections["boxes"]
            ):
                structured_results.append({
                    "index": object_counter,
                    "page": page_index + 1,
                    "image_index": img_index + 1,
                    "object_name": model.config.id2label[label.item()],
                    "coordinates": [round(v, 2) for v in box.tolist()],
                    "confidence": round(float(score), 4)
                })
                object_counter += 1

            # Save annotated image per page
            annotated = annotate_image(image.copy(), detections)
            annotated.save(f"page_{page_index+1}_image_{img_index+1}_annotated.png")

    return structured_results


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    results = process_pdf(PDF_PATH)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("Detection completed.")
    print(f"Structured results saved to {OUTPUT_JSON}")