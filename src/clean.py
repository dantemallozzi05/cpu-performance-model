import pandas as pd
import numpy as np

import os

def main():
    processed_dir = "data/processed"
    processed_file = "cpu_processed.csv"
    cleaned_file = "cpu_cleaned.csv"

    input_loc = os.path.join(processed_dir, processed_file)
    output_loc = os.path.join(processed_dir, cleaned_file)

    
