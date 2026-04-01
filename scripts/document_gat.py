import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer


class DocumentMultiTaskGAT(nn.Module):
    def __init__(self, num_node_classes=8):
        super(DocumentMultiTaskGAT, self).__init__()

        # Multimodálne projekcie
        self.geom_proj = nn.Linear(11, 64)
        self.yolo_proj = nn.Linear(11, 32)
        self.text_proj = nn.Linear(389, 160) # 5 manual + 384 transformer

        # GAT Chrbtica
        self.conv1 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(256, 32, heads=8, concat=True, dropout=0.2)

        # Hlavy
        self.node_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_node_classes)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        x_geom, x_yolo, x_text, edge_index = batch.feat_geom, batch.feat_yolo, batch.feat_text, batch.edge_index

        h_geom = F.elu(self.geom_proj(x_geom))
        h_yolo = F.elu(self.yolo_proj(x_yolo))
        h_text = F.elu(self.text_proj(x_text))
        x = torch.cat([h_geom, h_yolo, h_text], dim=-1)

        x_res1 = x
        x = F.elu(self.conv1(x, edge_index))
        x = x + x_res1

        x_res2 = x
        x = F.elu(self.conv2(x, edge_index))
        x = x + x_res2

        # Finálne embeddingy (tvoje 'z')
        z = F.elu(self.conv3(x, edge_index))
        node_logits = self.node_classifier(z)

        return z, node_logits

    def predict_edges(self, z, query_edge_index):
        # Vezme embeddingy zdrojových a cieľových uzlov
        src = z[query_edge_index[0]]
        dst = z[query_edge_index[1]]

        edge_features = torch.cat([src, dst], dim=-1)
        return self.edge_classifier(edge_features).squeeze(-1)


# ── 2. MAPOVANIE TRIED A POMOCNÉ FUNKCIE ──
GAT_CLASS_MAP = {
    0: "Caption", 1: "Picture", 2: "Table", 3: "Formula",
    4: "Section-header", 5: "Page-footer", 6: "Page-header", 7: "Other"
}


def extract_text_features(text: str) -> list:
    t = text.strip()
    tl = t.lower()

    is_fig = 1.0 if re.match(r'^(figure|fig\.?)\s*[\d\(\[]', tl) else 0.0
    is_tab = 1.0 if re.match(r'^(table|tab\.?)\s*[\d\(\[]', tl) else 0.0
    is_num = 1.0 if re.match(r'^\(?\d+[\.\)\:]', tl) else 0.0

    words = t.split()
    word_count = len(words)

    is_short = 1.0 if word_count < 20 else 0.0
    len_norm = min(word_count / 100.0, 1.0)

    return [is_fig, is_tab, is_num, is_short, len_norm]


def extract_features_for_gat_inference(page_nodes, text_embedder):
    if not page_nodes:
        return None, None, None

    raw_texts = [node.get("text", "") for node in page_nodes]

    # Hromadný text embedding
    embeddings = text_embedder.encode(raw_texts, convert_to_tensor=False, show_progress_bar=False)

    feat_geom, feat_yolo, feat_text = [], [], []

    for idx, node in enumerate(page_nodes):
        # 1. Geometrické features (11 dimenzií)
        x1_norm, y1_norm, x2_norm, y2_norm = node["geometry"]["normalized_coords"]
        x_center, y_center = node["geometry"]["normalized_center"]
        w_norm, h_norm = node["geometry"]["normalized_size"]

        area = np.log((w_norm * h_norm) + 1e-6)
        aspect = np.log((w_norm / (h_norm + 1e-6)) + 1e-6)
        conf = float(node.get("yolo_confidence", 1.0))

        geom_vector = [
            x1_norm, y1_norm, x2_norm, y2_norm,
            x_center, y_center, w_norm, h_norm, area, aspect, conf
        ]
        feat_geom.append(geom_vector)

        # 2. YOLO features (Soft labels - 11 dimenzií)
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

        # 3. Textové features (5 manuálnych + 384 transformer = 389 dimenzií)
        manual_text_feats = extract_text_features(raw_texts[idx])
        transformer_feats = embeddings[idx].tolist() if len(embeddings) > 0 else [0.0] * 384

        combined_text_vector = manual_text_feats + transformer_feats
        feat_text.append(combined_text_vector)

    return (
        torch.tensor(feat_geom, dtype=torch.float),
        torch.tensor(feat_yolo, dtype=torch.float),
        torch.tensor(feat_text, dtype=torch.float)
    )


def build_knn_edges_inference(features_geom, k=6):
    num_nodes = features_geom.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    actual_k = min(k, num_nodes - 1)

    # x_center a y_center sú na indexoch 4 a 5 v novom features_geom
    centers = features_geom[:, 4:6].numpy()
    edge_list = []

    for i in range(num_nodes):
        dx = centers[:, 0] - centers[i, 0]
        dy = centers[:, 1] - centers[i, 1]

        # Anizotropná vzdialenosť (rovnako ako v trénovaní)
        distances = np.sqrt(dx ** 2 + (1.5 * dy) ** 2)
        nearest = np.argsort(distances)[1:actual_k + 1]

        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])  # Pridávame oba smery

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)  # Odstránenie duplicít
    return edge_index


# ── 3. HLAVNÝ BEH (INFERENCIA) ──
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Načítanie modelov
    print("Načítavam GAT model...")
    model = DocumentMultiTaskGAT(num_node_classes=8).to(device)
    model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
    model.eval()

    print("Načítavam Text Embedder...")
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    json_path = "data/output_json.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        document_data = json.load(f)

    with torch.no_grad():
        for page in document_data["pages"]:
            if len(page["nodes"]) < 2:
                continue

            print(f"Processing page {page['page_number']}...")

            # Extrakcia príznakov
            feat_geom, feat_yolo, feat_text = extract_features_for_gat_inference(page["nodes"], text_embedder)

            # Vytvorenie hrán (K=6)
            edge_index = build_knn_edges_inference(feat_geom, k=6)

            # Vytvorenie dočasného "batch" objektu, aký GAT očakáva
            batch = Data(
                feat_geom=feat_geom.to(device),
                feat_yolo=feat_yolo.to(device),
                feat_text=feat_text.to(device),
                edge_index=edge_index.to(device)
            )

            # Forward pass
            z, node_logits = model(batch)

            # Výpočet pravdepodobností (softmax) a získanie najvyššej hodnoty a jej indexu
            node_probs = torch.softmax(node_logits, dim=-1)
            max_probs, node_preds = node_probs.max(dim=-1)

            # --- ULOŽENIE REKLASIFIKÁCIE DO JSONu ---
            node_preds = node_preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            for idx, node in enumerate(page["nodes"]):
                pred_class_id = int(node_preds[idx])

                node["gat_reclassified_id"] = pred_class_id
                node["gat_reclassified_label"] = GAT_CLASS_MAP.get(pred_class_id, "Unknown")
                node["gat_confidence"] = float(max_probs[idx])  # <--- TOTO pridaj

            # --- PREDIKCIA HRÁN ---
            edge_logits = model.predict_edges(z, batch.edge_index)
            edge_probs = torch.sigmoid(edge_logits)

            predicted_edges = []
            for i in range(len(edge_probs)):
                prob = edge_probs[i].item()

                # Znížený threshold (napr. 0.2), aby sa ti uložilo viacero hrán
                # (Caption -> Figure, Caption -> Table) a mohol si neskôr filtrovať
                if prob > 0.2:
                    src_idx = int(batch.edge_index[0, i].item())
                    dst_idx = int(batch.edge_index[1, i].item())

                    # Mapovanie späť na skutočné node_id z JSONu
                    src_node_id = page["nodes"][src_idx]["node_id"]
                    dst_node_id = page["nodes"][dst_idx]["node_id"]

                    predicted_edges.append({
                        "source": src_node_id,
                        "target": dst_node_id,
                        "probability": round(prob, 4)
                    })

            page["edges"] = predicted_edges
            print(f" -> Found {len(predicted_edges)} possible edges.")

    output_json = "data/results_hail_mary.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(document_data, f, ensure_ascii=False, indent=4)

    print(f"\nDone! JSON saved to: {output_json}")