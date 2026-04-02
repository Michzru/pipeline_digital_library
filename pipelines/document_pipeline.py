from stages.feature_engineering import run_feature_engineering
from stages.graph import run_graph_inference
from stages.text_detection import run_text_detection
from stages.yolo_detection import run_yolo_detection
from stages.preprocess_pdf import get_png_images
import json


def run_document_pipeline(pdf_path: str) -> dict:
    # Stage 0: Convert pdf to png
    pngs, filename = get_png_images(pdf_path)

    # Stage 1: YOLO detection
    yolo_detection_results = run_yolo_detection(pngs, filename)

    # Stage 2: OCR text detection
    text_yolo_detection_results = run_text_detection(pngs, yolo_detection_results)

    # Stage 3: Prepare graph document/Feature Engineering
    engineered_results = run_feature_engineering(text_yolo_detection_results)

    # Stage 4: graph (GAT)
    final_data = run_graph_inference(engineered_results)

    # Stage 5: save data
    with open(f"output_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, default=lambda x: None)

    return final_data