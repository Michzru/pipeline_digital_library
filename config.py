YOLO_MODEL_PATH = "model_weights/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
GAT_MODEL_PATH = "model_weights/best_model.pt"

TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_PIPELINE = "document"

CLASS_NAMES = {
    0: "Caption",
    1: "Picture",
    2: "Table",
    3: "Formula",
    4: "Section-header",
    5: "Page-footer",
    6: "Page-header",
    7: "Other",
}

UNITABLE_DIR = "models/unitable"
UNITABLE_STRUCTURE_WEIGHTS = "model_weights/unitable/unitable_large_structure.pt"
UNITABLE_BBOX_WEIGHTS = "model_weights/unitable/unitable_large_bbox.pt"
UNITABLE_CONTENT_WEIGHTS = "model_weights/unitable/unitable_large_content.pt"