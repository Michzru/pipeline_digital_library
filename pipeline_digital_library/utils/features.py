import torch
import re
import numpy as np
import unicodedata
from torch_geometric.data import Data

def remove_diacritics(text: str) -> str:
    """Normalizes text by removing diacritical marks"""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def extract_manual_text_features(text: str) -> list:
    """
        Creates 5 main text attributes
        Features:
          - is_fig: Is the text a figure caption? (EN/SK/CZ)
          - is_tab: Is the text a table caption? (EN/SK/CZ)
          - is_num: Does the text start with a number/equation?
          - is_short: Is the text short (likely a title/label)?
          - len_norm: Normalized word count (capped at 1.0)
    """
    t  = text.strip()
    tl = remove_diacritics(t.lower())

    # Figure captions (EN + SK + CZ)
    is_fig = 1.0 if re.match(
        r'^(figure|fig\.?|obrazok|obr\.?)\s*[\d\(\[]',
        tl
    ) else 0.0

    # Table captions (EN + SK + CZ)
    is_tab = 1.0 if re.match(
        r'^(table|tab\.?|tabulka)\s*[\d\(\[]',
        tl
    ) else 0.0

    # Numbered list / equation-like
    is_num = 1.0 if re.match(
        r'^\(?\d+[\.\)\:]',
        tl
    ) else 0.0

    words      = t.split()
    word_count = len(words)

    # Is it short like title?
    is_short   = 1.0 if word_count < 20 else 0.0
    len_norm   = min(word_count / 100.0, 1.0)

    return [is_fig, is_tab, is_num, is_short, len_norm]


def build_knn_edges(feat_geom, k=8):
    """
    Creates graph edges connected with k closest neighbors.
    Utilizes weighted Euclidean distance algorithm to compute the distance (prioritizes vertical relationships)
    """
    num_nodes = feat_geom.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    actual_k = min(k, num_nodes - 1)
    centers  = feat_geom[:, 4:6].numpy()

    edge_list = []
    for i in range(num_nodes):
        dx        = centers[:, 0] - centers[i, 0]
        dy        = centers[:, 1] - centers[i, 1]

        # Penalization of horizontal distance
        distances = np.sqrt(dx ** 2 + (1.5 * dy) ** 2)

        # Indices of k closest
        nearest   = np.argsort(distances)[1:actual_k + 1]

        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index

def prepare_page_tensors(page_nodes, text_embedder, image_W, image_H):
    """
    Takes nodes from one page and returns triplet tensors.
    Each node on the page:
      feat_geom  — [N, 11]  geometry + confidence
      feat_yolo  — [N, 11]  soft-label distribution of YOLO classes
      feat_text  — [N, 389] 5 hand-crafted + 384 transformer attributes
    """

    if not page_nodes:
        return None

    # Bulk embedding of texts (faster)
    texts = [node.get("text", "") for node in page_nodes]
    embeddings = text_embedder.encode(texts, convert_to_tensor=False,
                                      show_progress_bar=False)

    feat_geom, feat_yolo, feat_text = [], [], []

    for idx, node in enumerate(page_nodes):
        # Geometry (11 values)
        coords = node["geometry"]["absolute_pixel_coords"]
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1

        area = np.log(((w * h) / (image_W * image_H)) + 1e-6)
        aspect = np.log((w / (h + 1e-6)) + 1e-6)
        x_center = (x1 + w / 2) / image_W
        y_center = (y1 + h / 2) / image_H
        w_norm = w / image_W
        h_norm = h / image_H
        x1_norm = x1 / image_W
        y1_norm = y1 / image_H
        x2_norm = x2 / image_W
        y2_norm = y2 / image_H
        conf = float(node.get("yolo_confidence", 1.0))

        feat_geom.append([
            x1_norm, y1_norm, x2_norm, y2_norm,
            x_center, y_center, w_norm, h_norm,
            area, aspect, conf
        ])

        # YOLO soft-label (11), smooth label
        yolo_class_id = node.get("label_id", None)
        yolo_soft = np.full(11, (1.0 - conf) / 10.0, dtype=np.float32)

        if yolo_class_id is not None:
            class_idx = int(yolo_class_id)
            if 0 <= class_idx < 11:
                yolo_soft[class_idx] = conf
        else:
            yolo_soft = np.full(11, 1.0 / 11.0, dtype=np.float32)

        yolo_soft *= 0.3
        feat_yolo.append(yolo_soft.tolist())

        # Text features (5 + 384 = 389)
        manual_feats = extract_manual_text_features(texts[idx])
        transformer_feats = embeddings[idx].tolist()
        feat_text.append(manual_feats + transformer_feats)

    return (
        torch.tensor(feat_geom, dtype=torch.float),
        torch.tensor(feat_yolo, dtype=torch.float),
        torch.tensor(feat_text, dtype=torch.float),
    )