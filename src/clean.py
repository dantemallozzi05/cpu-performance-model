import pandas as pd
import numpy as np

import os

def main():
    # Hardcode location of data and file
    processed_dir = "data/processed"
    processed_file = "cpu_processed.csv"
    cleaned_file = "cpu_cleaned.csv"

    input_loc = os.path.join(processed_dir, processed_file)
    output_loc = os.path.join(processed_dir, cleaned_file)

    if not os.path.exists(input_loc):
        raise FileNotFoundError(f"Couldn't find processed input at: {input_loc}")
    
    # Load in pre-processed data to prepare for cleaning
    df = pd.read_csv(input_loc)

    # Load numeric features into DataFrame
    numeric_cols = ["price", "tdp", "speed", "turbo", "cpuCount", "cores", "logicals", "rank", "samples"]
    
    for col in numeric_cols:
        if col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill missing numeric values with median imputation
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # normalize confirmed numeric columns 
    present = [c for c in numeric_cols if c in df.columns]

    z = (df[present] - df[present].mean()) / df[present].std(ddof=0)

    # obtain general specification score for each cpu based on collective components
    df["spec_score"] = z.mean(axis=1)

    # classify into 4 tiers based on performative specs
    df["performance_tier"] = pd.qcut(
        df["spec_score"],
        q=4,
        labels=["low", "mid", "high", "exotic"]
    )

    # save newly cleaned dataset
    df.to_csv(output_loc, index=False)

    print(f"Saved newly cleaned dataset to {output_loc}")
    print(df["performance_tier"].value_counts())

if __name__ == "__main__":
    main()

