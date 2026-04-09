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

from pathlib import Path
from importlib.resources import files

PACKAGE_ROOT = Path(files("pipeline_digital_library").joinpath(""))
PROJECT_ROOT = PACKAGE_ROOT.parent

YOLO_MODEL_PATH = str(PROJECT_ROOT / "model_weights" / "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt")
GAT_MODEL_PATH = str(PROJECT_ROOT / "model_weights" / "best_model.pt")

UNITABLE_DIR = str(PACKAGE_ROOT / "models" / "unitable")

UNITABLE_STRUCTURE_WEIGHTS = str(PROJECT_ROOT / "model_weights" / "unitable" / "unitable_large_structure.pt")
UNITABLE_BBOX_WEIGHTS      = str(PROJECT_ROOT / "model_weights" / "unitable" / "unitable_large_bbox.pt")

UNITABLE_VOCAB_HTML = str(PACKAGE_ROOT / "models" / "unitable" / "vocab" / "vocab_html.json")
UNITABLE_VOCAB_BBOX = str(PACKAGE_ROOT / "models" / "unitable" / "vocab" / "vocab_bbox.json")