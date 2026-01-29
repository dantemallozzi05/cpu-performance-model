import os
import json
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import TransformerConv

from utils.gr_data import load_graph, make_split_masks
from utils.seed import seed_all

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            beta=True
        )

        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            beta=True
        )

    def forward(self, x, edge_index):
        x = self.con1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
# Helper method to evaluate model's performance
def eval_split(model, data, mask):
    model.eval()

    out = model(data.x, data.edge_index) 

    pred = out.argmax(dim=1)[mask].cpu().numpy() # model predictions
    obs = data.y[mask].cpu().numpy() # the actual true value

    acc = accuracy_score(obs, pred)
    f1 = f1_score(obs, pred, average="macro")

    return acc, f1

def main():
    seed_all(42)

    data = load_graph()
    train_mask, val_mask, test_mask = make_split_masks(data.y.cpu().numpy(), seed=42)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
