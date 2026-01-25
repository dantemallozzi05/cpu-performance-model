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
│   └───raw
├───notebooks
├───README.md 
├───results 
│   └───figures
├───src 
└───.gitignore
```

## Results


