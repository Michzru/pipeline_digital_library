from models.transformer import get_transformer_model
from utils.features import prepare_page_tensors, build_knn_edges

import torch

def run_feature_engineering(data):
    """
       Extracts features and builds the graph structure for each page.
       Does NOT run model inference or prediction.

       Args:
           data (dict): Document data containing pages and nodes.

       Returns:
           dict: Document data enriched with node features and edges (no predictions).
    """
    model = get_transformer_model()
    for page in data["pages"]:
        nodes = page["nodes"]

        if len(nodes) < 2:
            continue

        page_num = page["page_number"]
        image_W = page["width"]
        image_H = page["height"]

        print(f"Preprocessing page number {page_num}...")

        # Extraction of attributes
        result = prepare_page_tensors(nodes, model, image_W, image_H)

        if result is None:
            continue

        feat_geom, feat_yolo, feat_text = result

        # Graph Construction
        edge_index = build_knn_edges(feat_geom, k=8)

        # Store features and edges
        page["feat_geom"] = feat_geom
        page["feat_yolo"] = feat_yolo
        page["feat_text"] = feat_text
        page["edge_index"] = edge_index

    return data

