Load raw kaggle data to Data/next_day_wildfire_spread

src/data_processing/parse_data_to_pickle.py read the .tfrecord files and writes them to Data/processed and a small subset to Data/trial for testing purposes.

The pipeline src/feature_engineering.pipeline_feature_engineering.py executes feature engineering and data enrichment fuctions on the torch tensors saved in 
Data/processed and writes the enriched tensors to Data/feature_engineered.

# Overview:
ESC403_Project
│
├── Data
│   │
│   ├── next_day_wildfire_spread
│   │
│   ├── processed
│   │
│   ├── feature_engineered
│   │
│   └── trial
│
├── Notebooks
│
├── src
│   │
│   ├── data_processing
│   │
│   ├── feature_engineering
│   │
│   └── model_training
│
├── .gitignore
├── ESC403_Project.yml
├── config.py
└── utils.py
