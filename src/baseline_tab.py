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
    cleaned_file = "data/processed/cpu_cleaned.csv"

    if not os.path.exists(cleaned_file):
        raise ValueError(f"Couldn't find cleaned dataset at {cleaned_file}")
    
    # Load cleaned set and define target feature to model
    df = pd.read_csv(cleaned_file)

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

    print("Numeric Columns:", numeric_cols)
    print("Categorical Columns:", categorical_cols)

    # Initialize processing pipeline for numeric data 
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # pipeline for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # apply transforms to feature sets
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # declare baseline model 
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        ),
    }


    for model_name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        predict = pipe.predict(X_test)

        # Calculate Accuracy and F1 Scores, as well as a classification report
        acc = accuracy_score(y_test, predict)
        f1 = f1_score(y_test, predict, average="macro")

        print(f"Model: {model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print(classification_report(y_test, predict))

if __name__ == "__main__":
    main()
