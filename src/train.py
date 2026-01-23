import os
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

def make_split_mask(y):
    pass

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




if __name__ == "__main__":
    main()