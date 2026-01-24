import os
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


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

    train_mask[train] = True

    return train, val, test


def main():

    # define graph variables
    graph_dir = "data/processed/graph"
    x_loc = os.path.join(graph_dir, "x.npy")
    y_loc = os.path.join(graph_dir, "y.npy")
    e_loc = os.path.join(graph_dir, "i_edge.npy")

    X = np.load(x_loc).astype(np.float32)
    y = np.load(y_loc).astype(np.int64)
    i_edge = np.load(e_loc).astype(np.int64)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    i_edge = torch.tensor(i_edge, dtype=torch.long)

    train_mask, val_mask, test_mask = make_split_mask(y.numpy())

    # define data
    data = Data(
        X=X,
        y=y,
        i_edge=i_edge,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # initialize GraphSage model

    num_classes = int(y.max().item() + 1) 

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

        out = model(data.x, data.i_edge)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, train_f1 = eval_split(model, data, data.train_mask)
        val_acc, val_f1 = eval_split(model, data, data.val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        test_acc, test_f1 = eval_split(model, data, data.test_mask)
        print(f"Best val accuracy: {best_val_acc}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Test Macro F1: {test_f1}")



if __name__ == "__main__":
    main()