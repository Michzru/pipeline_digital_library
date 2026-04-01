from stages.detection import run_detection

#from ..stages.graph import run_graph
#from ..stages.extraction import run_extraction

from utils.preprocess_pdf import get_png_images

def run_document_pipeline(pdf_path: str) -> dict:
    # Step 0: Convert pdf to png
    pngs, filename = get_png_images(pdf_path)

    # Stage 1: detection
    detection_result = run_detection(pngs, filename)

    # # Stage 2: graph (GAT)
    # graph_result = run_graph(detection_result)
    #
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