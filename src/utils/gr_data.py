import os 
import numpy as np 
import torch

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

def load_graph(graph_dir="data/processed/graph"):
    x = np.load(os.path.join(graph_dir, "x.npy")).astype(np.float32)
    y = np.load(os.path.join(graph_dir, "y.npy")).astype(np.int64)
    edge = np.load(os.path.join(graph_dir, "i_edge.npy")).astype(np.int64)

    x_t = torch.tensor(x, dtype=torch.float32),
    y_t = torch.tensor(y, dtype=torch.long),
    e_t = torch.tensor(e, dtype=torch.long)

    data = Data(
        x=x_t,
        y=y_t,
        edge_index=e_t
    )
    return data


def make_split_masks(y, seed=42):
    id_n = np.arrange(len(y))

    train, test = train_test_split(
        id_n, test_size=0.2, random_state=seed, stratify=y
    )

    train, val = train_test_split(
        train, test_size=0.2, random_state=seed, stratify=y[train]
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train]=True
    val_mask[val]=True
    test_mask[test]=True

    return train_mask, val_mask, test_mask


