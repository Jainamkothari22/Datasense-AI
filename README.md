# DataSense AI

A portfolio-ready Streamlit workspace for data analysts and data scientists. Upload a CSV or Excel file, inspect data quality, apply reproducible cleaning steps, build charts, transform columns, and train a measurable baseline model.

## Features

- CSV and Excel upload
- Multi-file upload with append and relational join workflows
- Relationship analysis across datasets using shared key columns and inner, left, right, or outer joins
- Dataset profiling: schema, missingness, uniqueness, duplicates
- Cleaning recipe: normalize headers, trim text, remove duplicates, impute or drop missing values, clip numeric outliers
- Visual exploration: histogram, bar, line, scatter, box summaries, correlation matrix
- Transformations: filter, rename, type conversion, sorting, sampling
- Baseline ML: automatic classification/regression detection, preprocessing pipeline, random forest or linear/logistic model, accuracy/R2 and test predictions
- SQLite project and activity history
- Downloadable working dataset

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, then upload a dataset.

## Portfolio talking points

This project shows an end-to-end analytics workflow rather than a single notebook: data ingestion, data quality, transformation, visualization, reproducible ML preprocessing, evaluation, persistence, and a user-facing interface. For a production version, add authentication, cloud object storage, job queues, experiment tracking, automated tests, and model monitoring.
