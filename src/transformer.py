import os
import json
import numpy as np
import torch
import torch.nn.functional as F

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
        x = self.conv1(x, edge_index)
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

    num_classes = int(data.y.unique().numel())

    model = GraphTransformer(
        in_channels=data.num_features,
        hidden_channels=32,
        out_channels=num_classes,
        heads=4,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=5e-4
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(0, 200):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc, val_f1 = eval_split(model, data, data.val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_acc, test_f1 = eval_split(model, data, data.test_mask)

    results = {
        "model": "GraphTransformer",
        "accuracy": float(test_acc),
        "macro_f1": float(test_f1)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/transf_results.json", "w") as file:
        json.dump(results, file, indent=2)

    print(results)

if __name__ == "__main__":
    main()