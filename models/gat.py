from config import GAT_MODEL_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

_model = None

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


def get_gat_model(verbose, gpu = True):
    global _model

    if _model is None:
        if gpu:
            _model = DocumentMultiTaskGAT(num_node_classes=8).to('cuda')
            _model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location='cuda'))
            _model.eval()
            if verbose:
                print("GAT model loaded on GPU.")
        else:
            _model = DocumentMultiTaskGAT(num_node_classes=8).to('cpu')
            _model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location='cpu'))
            _model.eval()
            if not verbose:
                print("GAT model loaded on CPU.")

    return _model