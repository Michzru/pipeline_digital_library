import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL  (identický s tréningom)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMultiTaskGAT(nn.Module):
    def __init__(self, num_node_classes=8):
        super().__init__()

        self.geom_proj = nn.Linear(11, 64)
        self.yolo_proj = nn.Linear(11, 32)
        self.text_proj = nn.Linear(389, 160)   # 5 manual + 384 transformer

        self.conv1 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(256, 32, heads=8, concat=True, dropout=0.2)

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
        x_geom  = batch.feat_geom
        x_yolo  = batch.feat_yolo
        x_text  = batch.feat_text
        edge_index = batch.edge_index

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

        z = F.elu(self.conv3(x, edge_index))
        node_logits = self.node_classifier(z)

        return z, node_logits

    def predict_edges(self, z, query_edge_index):
        src = z[query_edge_index[0]]
        dst = z[query_edge_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        return self.edge_classifier(edge_features).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAPOVANIE TRIED
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRAKCIA FEATURES  (zjednotená s tréningom)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_features(text: str) -> list:
    """5 ručných textových príznakov — rovnaká funkcia ako v tréningu."""
    t  = text.strip()
    tl = t.lower()

    is_fig   = 1.0 if re.match(r'^(figure|fig\.?)\s*[\d\(\[]', tl) else 0.0
    is_tab   = 1.0 if re.match(r'^(table|tab\.?)\s*[\d\(\[]',  tl) else 0.0
    is_num   = 1.0 if re.match(r'^\(?\d+[\.\)\:]', tl)             else 0.0

    words      = t.split()
    word_count = len(words)
    is_short   = 1.0 if word_count < 20 else 0.0
    len_norm   = min(word_count / 100.0, 1.0)

    return [is_fig, is_tab, is_num, is_short, len_norm]


def extract_features_for_inference(page_nodes, text_embedder, image_W, image_H):
    """
    Pre každý uzol stránky vytvorí:
      feat_geom  — [N, 11]  geometria + confidence
      feat_yolo  — [N, 11]  soft-label distribúcia YOLO triedy
      feat_text  — [N, 389] 5 ručných + 384 transformer príznakov

    Vracia trojicu tensorov, alebo None ak je stránka prázdna.
    Logika je identická s build_node_features() v tréningovom kóde.
    """
    if not page_nodes:
        return None

    # Hromadný embedding textov (rýchlejšie ako po jednom)
    texts      = [node.get("text", "") for node in page_nodes]
    embeddings = text_embedder.encode(texts, convert_to_tensor=False,
                                      show_progress_bar=False)

    feat_geom, feat_yolo, feat_text = [], [], []

    for idx, node in enumerate(page_nodes):
        coords        = node["geometry"]["absolute_pixel_coords"]
        x1, y1, x2, y2 = coords
        w, h          = x2 - x1, y2 - y1

        # ── Geometrické príznaky (11) ──────────────────────────────────────
        area     = np.log(((w * h) / (image_W * image_H)) + 1e-6)
        aspect   = np.log((w / (h + 1e-6)) + 1e-6)
        x_center = (x1 + w / 2) / image_W
        y_center = (y1 + h / 2) / image_H
        w_norm   = w / image_W
        h_norm   = h / image_H
        x1_norm  = x1 / image_W
        y1_norm  = y1 / image_H
        x2_norm  = x2 / image_W
        y2_norm  = y2 / image_H
        conf     = float(node.get("yolo_confidence", 1.0))

        feat_geom.append([
            x1_norm, y1_norm, x2_norm, y2_norm,
            x_center, y_center, w_norm, h_norm,
            area, aspect, conf
        ])

        # ── YOLO soft-label (11) ───────────────────────────────────────────
        yolo_class_id = node.get("label_id", None)
        yolo_soft     = np.full(11, (1.0 - conf) / 10.0, dtype=np.float32)

        if yolo_class_id is not None:
            class_idx = int(yolo_class_id)
            if 0 <= class_idx < 11:
                yolo_soft[class_idx] = conf
        else:
            yolo_soft = np.full(11, 1.0 / 11.0, dtype=np.float32)

        yolo_soft *= 0.3          # škálovanie rovnaké ako v tréningu
        feat_yolo.append(yolo_soft.tolist())

        # ── Textové príznaky (5 + 384 = 389) ──────────────────────────────
        manual_feats      = extract_text_features(texts[idx])
        transformer_feats = embeddings[idx].tolist()
        feat_text.append(manual_feats + transformer_feats)

    return (
        torch.tensor(feat_geom, dtype=torch.float),
        torch.tensor(feat_yolo, dtype=torch.float),
        torch.tensor(feat_text, dtype=torch.float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. KNN HRANY  (zjednotené s tréningom)
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_edges_inference(feat_geom, k=8):
    """
    x_center a y_center sú na indexoch 4 a 5 feat_geom vektora
    (rovnaká logika ako build_knn_edges() v tréningovom kóde).
    """
    num_nodes = feat_geom.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    actual_k = min(k, num_nodes - 1)
    centers  = feat_geom[:, 4:6].numpy()   # x_center, y_center

    edge_list = []
    for i in range(num_nodes):
        dx        = centers[:, 0] - centers[i, 0]
        dy        = centers[:, 1] - centers[i, 1]
        distances = np.sqrt(dx ** 2 + (1.5 * dy) ** 2)
        nearest   = np.argsort(distances)[1:actual_k + 1]

        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# ─────────────────────────────────────────────────────────────────────────────
# 5. FILTROVANIE CAPTION HRÁN
# ─────────────────────────────────────────────────────────────────────────────

def filter_caption_edges(predicted_edges, node_predicted_classes):
    """
    Pre každý Caption uzol (trieda 0) zachová len hranu s najvyššou
    pravdepodobnosťou. Ostatné uzly nie sú dotknuté.

    node_predicted_classes: dict  {node_id: class_id}
    """
    # Zozbierame najlepšiu hranu pre každý caption uzol
    caption_best: dict = {}   # caption_node_id -> edge dict
    non_caption_edges = []

    for edge in predicted_edges:
        src, dst    = edge["source"], edge["target"]
        src_caption = (node_predicted_classes.get(src) == 0)
        dst_caption = (node_predicted_classes.get(dst) == 0)

        if src_caption or dst_caption:
            # Ak sú caption uzlom oba, berie sa src ako kľúč (prípad je zriedkavý)
            cap_node = src if src_caption else dst
            if (cap_node not in caption_best
                    or edge["probability"] > caption_best[cap_node]["probability"]):
                caption_best[cap_node] = edge
        else:
            non_caption_edges.append(edge)

    return non_caption_edges + list(caption_best.values())


# ─────────────────────────────────────────────────────────────────────────────
# 6. HLAVNÝ INFERENČNÝ BLOK
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Načítavam sentence-transformer...")
text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

print("Načítavam model...")
model = DocumentMultiTaskGAT(num_node_classes=8).to(device)
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.eval()
print("Model pripravený.\n")

json_path = "data/output_json.json"
with open(json_path, "r", encoding="utf-8") as f:
    document_data = json.load(f)

with torch.no_grad():
    for page in document_data["pages"]:
        nodes = page["nodes"]
        if len(nodes) < 2:
            continue

        page_num = page["page_number"]
        image_W  = page["width"]
        image_H  = page["height"]
        print(f"Spracovávam stránku {page_num}...")

        # ── Extrakcia príznakov ───────────────────────────────────────────
        result = extract_features_for_inference(nodes, text_embedder,
                                                image_W, image_H)
        if result is None:
            continue

        feat_geom, feat_yolo, feat_text = result
        edge_index = build_knn_edges_inference(feat_geom, k=8)

        feat_geom  = feat_geom.to(device)
        feat_yolo  = feat_yolo.to(device)
        feat_text  = feat_text.to(device)
        edge_index = edge_index.to(device)

        # ── Vytvoríme Data objekt (rovnaká štruktúra ako v tréningu) ──────
        batch = Data(
            feat_geom  = feat_geom,
            feat_yolo  = feat_yolo,
            feat_text  = feat_text,
            edge_index = edge_index,
        ).to(device)

        # ── Inferencia ────────────────────────────────────────────────────
        z, node_logits = model(batch)
        pred_classes   = node_logits.argmax(dim=-1)   # [N]
        node_probs = torch.softmax(node_logits, dim=-1)

        # ── Zapíšeme reklasifikovanú triedu späť do uzlov ─────────────────
        node_predicted_classes = {}
        for i, node in enumerate(nodes):
            class_id = int(pred_classes[i].item())
            class_name = CLASS_NAMES[class_id]
            node["predicted_label"] = class_name
            node["predicted_label_id"] = class_id
            node["predicted_confidence"] = round(float(node_probs[i, class_id].item()), 4)  # ← PRIDAJ
            node_predicted_classes[node["node_id"]] = class_id

        # ── Predikcia hrán ────────────────────────────────────────────────
        edge_logits = model.predict_edges(z, edge_index)
        edge_probs  = torch.sigmoid(edge_logits)

        raw_edges = []
        for i in range(len(edge_probs)):
            prob = edge_probs[i].item()
            if prob > 0.3:
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())
                raw_edges.append({
                    "source": src,
                    "target": dst,
                    "probability": round(prob, 4),  # toto už máš
                    "confidence": round(prob, 4),  # ← PRIDAJ ak chceš explicitný kľúč
                })

        # ── Filtrovanie caption hrán (len 1 najlepšia na caption uzol) ────
        #filtered_edges = filter_caption_edges(raw_edges, node_predicted_classes)
        filtered_edges = raw_edges

        page["edges"] = filtered_edges
        print(f"  -> {len(nodes)} uzlov reklasifikovaných, "
              f"{len(raw_edges)} raw hrán → {len(filtered_edges)} po filtrovaní.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ULOŽENIE VÝSTUPU
# ─────────────────────────────────────────────────────────────────────────────

output_json = "data/document_graph_with_edges_test.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(document_data, f, ensure_ascii=False, indent=4)

print(f"\nHotovo! JSON s hranami a reklasifikovanými uzlami: {output_json}")