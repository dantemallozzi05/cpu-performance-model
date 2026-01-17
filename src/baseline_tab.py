import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Baseline model: trained with Logistic Regression and Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    # Hardcode path and verify cleaned data is found
    input_loc = "data/processed/cpu_cleaned.csv"

    if not os.path.exists(input_loc):
        raise ValueError(f"Couldn't find cleaned dataset at {input_loc}")
    
    # Load cleaned set and define target feature to model
    df = pd.read_csv(input_loc)

    target_col = "performance_tier"
    y = df[target_col].astype(str)

    # Remove name from feature columns
    drop_cols = [target_col]
    if "name" in df.columns:
        drop_cols.append("name")

    X = df.drop(columns=drop_cols, errors="ignore")

    # split columns into numeric and categorical groups
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()



if __name__ == "__main__":
    main()
