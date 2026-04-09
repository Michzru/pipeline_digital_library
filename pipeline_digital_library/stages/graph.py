from ..models.gat import get_gat_model
import torch
from torch_geometric.data import Data
from config import CLASS_NAMES
from tqdm import tqdm

def run_graph_inference(data, verbose, gpu):
    model = get_gat_model(verbose=verbose, gpu=gpu)

    if gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model.eval()

    pages = data["pages"]

    iterator = tqdm(
        enumerate(pages),
        total=len(pages),
        desc="Graph inference",
        disable=not verbose,
        leave=False
    )

    with torch.no_grad():
        for page_idx, page in iterator:
            if "feat_geom" not in page:
                continue

            if verbose:
                iterator.set_postfix(page=page_idx + 1)

            feat_geom = torch.as_tensor(page["feat_geom"], dtype=torch.float32)
            feat_yolo = torch.as_tensor(page["feat_yolo"], dtype=torch.float32)
            feat_text = torch.as_tensor(page["feat_text"], dtype=torch.float32)
            edge_index = torch.as_tensor(page["edge_index"], dtype=torch.long)

            batch = Data(
                feat_geom=feat_geom,
                feat_yolo=feat_yolo,
                feat_text=feat_text,
                edge_index=edge_index
            ).to(device)

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
                node["predicted_confidence"] = round(float(node_probs[i, class_id].item()), 4)
                node_predicted_classes[node["node_id"]] = class_id

            # Prediction of edges
            edge_logits = model.predict_edges(z, edge_index)
            edge_probs = torch.sigmoid(edge_logits)

            page["edges"] = []
            for i in range(len(edge_probs)):
                if edge_probs[i] > 0.5:
                    page["edges"].append({
                        "source": int(edge_index[0, i].item()),
                        "target": int(edge_index[1, i].item()),
                        "confidence": float(edge_probs[i].item())
                    })
            page.pop("feat_geom", None)
            page.pop("feat_yolo", None)
            page.pop("feat_text", None)
            page.pop("edge_index", None)
    return data