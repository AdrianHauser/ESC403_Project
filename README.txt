Load Data

src/data_processing/pipeline_data_preprocessing: Loads data from Kaggle to Data/raw. Reads it to torch tensors and saves it in preprocessed in .pkl format. Also writes a small trial-subset to Data/trial for testing purposes.

The pipeline src/feature_engineering.pipeline_feature_engineering.py executes feature engineering and data enrichment fuctions on the torch tensors saved in 
Data/preprocessed and writes the enriched tensors to Data/feature_engineered again in .pkl format.

All modelling was done in Jupyter Notebooks stored in /Notebooks. Most models were trained in Google Colab to use GPU's.

# Overview:
ESC403_Project
│
├── Data
│   │
│   ├── raw
│   │
│   ├── preprocessed
│   │
│   ├── feature_engineered
│   │
│   └── trial
│
├── Notebooks
│
├── src
│   │
│   ├── data_preprocessing
│   │
│   ├── feature_engineering
│   │
│   └── model_training
│
├── .gitignore
├── ESC403_Project.yml
├── config.py
└── utils.py
