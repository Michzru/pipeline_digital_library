from models.gat import get_gat_model
import torch
from torch_geometric.data import Data
from config import CLASS_NAMES

def run_graph_inference(data):
    model = get_gat_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        for page in data["pages"]:
            if "feat_geom" not in page:
                continue

            batch = Data(
                feat_geom=page["feat_geom"].to(device),
                feat_yolo=page["feat_yolo"].to(device),
                feat_text=page["feat_text"].to(device),
                edge_index=page["edge_index"].to(device)
            )

            # Inference
            z, node_logits = model(batch)

            # Prediction of node classes (reclassification)
            pred_classes = node_logits.argmax(dim=-1)
            node_probs = torch.softmax(node_logits, dim=-1)

            node_predicted_classes = {}
            for i, node in enumerate(page["nodes"]):
                class_id = int(pred_classes[i].item())
                class_name = CLASS_NAMES[class_id]
                node["predicted_label"] = class_name
                node["predicted_label_id"] = class_id
                node["predicted_confidence"] = round(float(node_probs[i, class_id].item()), 4)  # ← PRIDAJ
                node_predicted_classes[node["node_id"]] = class_id

            # Prediction of edges
            edge_logits = model.predict_edges(z, page["edge_index"].to(device))
            edge_probs = torch.sigmoid(edge_logits)

            page["edges"] = []
            for i in range(len(edge_probs)):
                if edge_probs[i] > 0.5:
                    page["edges"].append({
                        "source": int(page["edge_index"][0, i].item()),
                        "target": int(page["edge_index"][1, i].item()),
                        "confidence": float(edge_probs[i].item())
                    })

    return data