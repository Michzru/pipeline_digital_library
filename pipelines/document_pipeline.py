from stages.text_detection import run_text_detection
from stages.yolo_detection import run_yolo_detection
from stages.preprocess_pdf import get_png_images

#from ..stages.graph import run_graph
#from ..stages.extraction import run_extraction



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
    #graph_result = run_graph(detection_result)

    # # Stage 3: extraction
    # extraction_result = run_extraction(graph_result)
    #
    # return {
    #     "status": "success",
    #     "stages": {
    #         "detection": detection_result,
    #         "graph": graph_result,
    #         "extraction": extraction_result
    #     },
    #     "final_output": extraction_result
    # }
    return True