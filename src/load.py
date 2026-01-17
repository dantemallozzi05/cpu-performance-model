import pandas as pd
import os

def main():

    # Hardcode location of data with path
    data_dir = "data/raw"
    processed_dir = "data/processed"
    data_file = "cpu_data.csv"
    processed_file = "cpu_processed.csv"

    data_loc = os.path.join(data_dir, data_file)

    if not os.path.exists(data_loc):
        raise FileNotFoundError(f"Couldn't locate in {data_loc}")
    
    # Load Data
    df = pd.read_csv(data_loc)

    # Early analysis
    print(df.shape)
    print(df.columns.tolist())

    # Saving process copy
    processed_loc = os.path.join(processed_dir, processed_file)
    df.to_csv(processed_loc, index=False)

    print(f"Saved processed copy in {processed_loc}")

if __name__ == "__main__":
    main()
