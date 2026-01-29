import os
import numpy as np
import torch
import torch.nn.functional as F
import json

from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.nn import SAGEConv

# custom helper methods for data processing
from utils.gr_data import load_graph, make_split_masks
from utils.seed import seed_all

# Implementing GNN model GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

def eval_split(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)[mask].cpu().numpy()
    true = data.y[mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    return acc, f1

def main():

    # Common seed 
    seed_all(42)

    data = load_graph()
    train_mask, val_mask, test_mask = make_split_masks(data.y.cpu().numpy(), seed=42)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # initialize GraphSage model

    num_classes = int(data.y.unique().numel()) 

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=64,
        out_channels=num_classes
    ).to(device)

    # initialize adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )

    # training loop over 200 epochs
    best_val_acc = 0.0
    best_state = None

    for epoch in range(0, 200):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, train_f1 = eval_split(model, data, data.train_mask)
        val_acc, val_f1 = eval_split(model, data, data.val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_acc, test_f1 = eval_split(model, data, data.test_mask)
    print(f"Best val accuracy: {best_val_acc}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Macro F1: {test_f1}")
    results = {
        "model": "GraphSAGE",
        "accuracy": float(test_acc),
        "macro_f1": float(test_f1),
        "best_val_accuracy": float(best_val_acc)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/gnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(results) 

if __name__ == "__main__":
    main()