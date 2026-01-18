import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

def main():

    cleaned_file = "cpu_cleaned.csv"
    graph_dir = "data/processed/graph"

    os.makedirs(graph_dir, exist_ok=True)

    k = 10

    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Couldn't find cleaned dataset in {cleaned_file}")
    
    df = pd.read_csv(cleaned_file)

    target_col = "performance_tier"
    

    # Build labels for graph
    le = LabelEncoder()

    y = le.fit_transform(df[target_col].astype(str))
    label_map = {cls_name: int(idx) for idx, cls_name in enumerate(le.classes_)} 

    # add numeric features to nodes & normalize
    drop_cols = [target_col]
    if "name" in df.columns:
        drop_cols.append("name")

    X_df = df.drop(columns=drop_cols, errors="ignore")

    numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()

    X = X_df[numeric_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    

