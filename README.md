# Computer Vision: PDF Construction Image Object Detection

## Description
This project focuses on object detection in images extracted from PDF files, specifically for construction-related images. It leverages Hugging Face Transformers (DETR and Grounding DINO) to detect objects such as trucks, cars, hard hats, excavators, and more.

## Main Features
- Extract images from PDF (ignoring text)
- Automatic object detection using pretrained models
- Filter detection results by confidence threshold
- Output as structured JSON and annotated images
- Example for zero-shot object detection and hyperparameter fine-tuning

## File Structure
- `task_file.py`: Main pipeline for PDF image extraction and object detection (DETR)
- `examples.py`: Example of zero-shot object detection with Grounding DINO
- `example_hyperparameter.py`: Example of hyperparameter settings for fine-tuning
- `detection_results.json`: Example of general object detection results
- `construction_detection_results.json`: Example of construction image detection results
- In this current result, dataset is coming from https://stgenpln.blob.core.windows.net/document/Permanente_20141031_MRRC.pdf.

## How to Run
1. **Install Dependencies**
   ```bash
   pip install torch transformers pillow pymupdf datasets
   ```
2. **Run Detection on PDF**
   - Edit `PDF_PATH` in `task_file.py` to your PDF file
   - Run:
     ```bash
     python task_file.py
     ```
   - Results will be saved in `construction_detection_results.json` and annotated images

3. **Zero-Shot Detection Example**
   - Make sure you have an image file (e.g., `new_sample.png`)
   - Run:
     ```bash
     python examples.py
     ```

4. **Fine-tuning (Optional)**
   - Edit and use `example_hyperparameter.py` for your custom dataset

## Notes
- Default model: `facebook/detr-resnet-50` (spatial-aware)
- For zero-shot: `IDEA-Research/grounding-dino-base`
- Detection result JSON contains: object name, bounding box coordinates, confidence, page, etc.
