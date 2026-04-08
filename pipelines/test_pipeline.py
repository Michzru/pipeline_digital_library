from stages.table_extraction import run_table_extraction
import numpy as np
from stages.feature_engineering import run_feature_engineering
from stages.graph import run_graph_inference
from stages.text_detection import run_text_detection
from stages.yolo_detection import run_yolo_detection
from stages.preprocess_pdf import get_png_images
from tqdm import tqdm
import json
import json
import os

CACHE_DIR = "D:\\Bakalarka\\pipeline_digital_library\\data\\cache\\pipeline"

def save_stage(data, filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=convert)

def convert(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def load_stage(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_test_pipeline(pdf_path: str, verbose = True, gpu=True) -> dict:
    # Stage 0: Convert pdf to png
    #pngs, filename = get_png_images(pdf_path, verbose)

    from utils.cache import load_png_images
    import os
    # 1. Získaj absolútnu cestu ku koreňu tvojho projektu
    BASE_DIR = "D:\\Bakalarka\\pipeline_digital_library\\data\cache"

    full_cache_path = os.path.join(BASE_DIR, pdf_path.replace('.pdf',''))
    filename= 'dbbf8dde-1d40-481b-ba5f-4d84c2de3e54'

    pngs = load_png_images(full_cache_path)

    print("cache loaded")

    # Stage 1: YOLO
    yolo_detection_results = load_stage(f"{filename}_yolo.json")
    if yolo_detection_results is None:
        yolo_detection_results = run_yolo_detection(pngs, filename, verbose, gpu)
        save_stage(yolo_detection_results, f"{filename}_yolo.json")

    # Stage 2: OCR
    text_yolo_detection_results = load_stage(f"{filename}_ocr.json")
    if text_yolo_detection_results is None:
        text_yolo_detection_results = run_text_detection(pngs, yolo_detection_results, verbose, gpu)
        save_stage(text_yolo_detection_results, f"{filename}_ocr.json")

    # Stage 3: Feature Engineering
    engineered_results = load_stage(f"{filename}_engineered.json")
    if engineered_results is None:
        engineered_results = run_feature_engineering(text_yolo_detection_results, verbose, gpu)
        save_stage(engineered_results, f"{filename}_engineered.json")

    # Stage 4: GAT
    graph_data = load_stage(f"{filename}_graph.json")
    if graph_data is None:
        graph_data = run_graph_inference(engineered_results, verbose, gpu)
        save_stage(graph_data, f"{filename}_graph.json")

    # Stage 5: table extraction
    final_data = load_stage(f"{filename}_final.json")
    if final_data is None:
        final_data = run_table_extraction(graph_data, pngs, verbose, gpu)
        save_stage(final_data, f"{filename}_final.json")


    # Stage 6: save data [optional]
    # with open(f"data/outputs/output_{filename}.json", "w", encoding="utf-8") as f:
    #     json.dump(final_data, f, indent=4, default=lambda x: None)

    #return final_data