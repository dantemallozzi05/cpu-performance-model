# Graph-Based CPU Performance Modeling

This project explores the different **graph-based machine learning** approaches for modeling CPU performance using hardware specification data.

Each CPU is stored as a node in a similarity graph, with edges connecting processors that share comparable archiectural characteristics.

## Project Goals
- Model CPU performance metrics using relational structure rather than isolated components
- Compare tabular ML models with Graph Neural Networks (GNU, GraphSAGE)
- Build a reproducible, end-to-end ML pipeline suitable for systemic applications

## Dataset
CPU specification data sourced from Kaggle (located in `data/raw/`).

## Project Structure
``` bash
├───data
│   ├───processed
│   │   └───graph
│   └───raw
├───notebooks
├───results
│   └───figures
└───src
    └───utils
```

## Pipeline Overview

1. Raw CPU specifications are loaded / cleaned
2. Numeric features are standardized
3. Performance tiers are derived as proxy labels
4. kNN similarity graph with cosine similarity
5. Models trained with train/val/test splits
6. Results are parsed to JSON for notebook analysis

## Results

| Model                      | Accuracy | Macro F1 |
| -------------------------- | -------- | -------- |
| Logistic Regression        | 0.9473   | 0.9476   |
| Random Forest              | 0.9899   | 0.9899   | 
| GraphSAGE                  | 0.9787   | 0.9787   | 
| Graph Transformer          | 0.9585   | 0.9587   |

## Discussion
Random Forest outperforms graph-based models, due to strong nonlinear signal existing in tabular features. However, GraphSAGE improves over Logistic Regression, indicating that relational structure provides useful inductive bias.

Since performance tiers are derived from input features, this task is framed as a proxy classification problem, inherently favoring tabular models. Graph-based approaches are expected to provide greater benefits when features are incomplete or when relationships are not effectively captured in tabular form.

## Reproducibility

All experiments conducted are fully reproducible.

1. Place raw dataset in `data/raw/`
2. Run preprocessing and training scripts
3. Generated results are saved to `results/`