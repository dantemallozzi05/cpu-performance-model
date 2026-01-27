import os
import json
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():

    graph_dir = "data/processed/graph"
    x_loc = os.path.join(graph_dir, "x.npy")
    y_loc = os.path.join(graph_dir, "y.npy")
    e_loc = os.path.join(graph_dir, "i_edge.npy")
    
    X = np.load(x_loc).astype(np.float32)
    y = np.load(y_loc).astype(np.int64)
    i_edge = np.load(e_loc).astype(np.int64)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    i_edge_t = torch.tensor(edge_index, dtype=torch.long)

    data = Data(

    )
