import os
import numpy as np
import torch
import torch.nn.functional as F
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

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
    
# make_split_mask - assigns nodes into train, evaluation, and test groups
def make_split_mask(y):
    id_n = np.arange(len(y))

    train, test = train_test_split(
        id_n, test_size=0.2, stratify=y
    )

    train, val = train_test_split(
        train, test_size=0.2, stratify=y[train]
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train] = True
    val_mask[val] = True
    test_mask[test] = True

    return train, val, test

def eval_split(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)[mask].cpu().numpy()
    true = data.y[mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    return acc, f1

def main():

    # define graph variables
    graph_dir = "data/processed/graph"
    x_loc = os.path.join(graph_dir, "x.npy")
    y_loc = os.path.join(graph_dir, "y.npy")
    e_loc = os.path.join(graph_dir, "i_edge.npy")

    X = np.load(x_loc).astype(np.float32)
    y = np.load(y_loc).astype(np.int64)
    i_edge = np.load(e_loc).astype(np.int64)

    X_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    i_edge_t = torch.tensor(i_edge, dtype=torch.long)

    train_mask, val_mask, test_mask = make_split_mask(y)

    # define data
    data = Data(
        x=X_t,
        y=y_t,
        edge_index=i_edge_t,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # initialize GraphSage model

    num_classes = int(np.unique(y).size) 

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
        "model": "Graph",
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