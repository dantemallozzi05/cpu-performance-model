import pandas as pd
import os

def main():

    # Hardcode location of data with path
    data_dir = "data/raw"
    processed_dir = "data/processed"
    data_file = "cpu_data.csv"

    
    data_loc = os.path.join(data_dir, data_file)
    processed_loc = os.path.join(processed_dir, data_file)

    if not os.path.exists(data_loc):
        raise FileNotFoundError(f"Couldn't locate in {data_loc}")

    




main()
