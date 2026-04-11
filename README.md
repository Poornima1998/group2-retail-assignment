# Member 1 - ETL & Data Preprocessing

This repository contains my individual contribution for the group retail analytics 
assignment.

## My Role
Member 1 - ETL & Data Preprocessing

## Tasks Completed
- Extracted and loaded all raw datasets
- Cleaned missing values, inconsistencies, and invalid entries
- Parsed date columns- Standardized categorical values
- Handled duplicates
- Capped outliers using the IQR method
- Performed feature engineering- Generated cleaned datasets
- Built an integrated customer master table
- Documented ETL outputs in summary and JSON report files

## Main Files
- `scripts/etl.py` - ETL pipeline script
- `notebooks/01_etl_preprocessing.ipynb` - ETL notebook
- `data/raw/` - original input datasets
- `data/processed/` - cleaned datasets
- `reports/etl_report.json` - detailed ETL report
- `reports/etl_summary.txt` - ETL summary

## How to Run
```bash
pip install -r requirements.txt
python scripts/etl.py
```
## Outputs
Running the ETL script generates:
- cleaned CSV files in `data/processed/`
- ETL reports in `reports/`

## Tools Used
- Python
- Pandas
- NumPy
- Jupyter Notebook

# Member 2 - Exploratory Data Analysis

Detailed data analysis with descriptive statistics with insightful visualization

## How to Run
```bash
python scripts/eda.py
```

## Analysis based on
- focused on sales trends, payment behavior, interaction channels, campaign performance, review ratings, and support operations.

## Main Files
- `scripts/eda.py` - EDA script
- `notebooks/Etl_Analysis.ipynb` - EDA notebook
- `reports/eda_summary.txt` - ETL summary
  
## Outputs
Running the EDA script generates:
- descriptive tables in `outputs/tables/`
- visualizations in `outputs/plots/`
- insight reports in `reports/insights_summary.txt`

## Tools Used
- Python
- Pandas
- NumPy
- matplotlib
- Jupyter Notebook

# Member 3 - Clustering Analysis

Implement machine learning technique to segment the customer base into distinct behaviour profiles, actionable insights for efficiency and targeted marketing.

## Tasks Completed.
- Feature Engineering & Selection
- Data Normalization
- Data Preprocessing
- Dimensionality Reduction
- Cluster Profiling
- Automated Reporting

## How to Run
```bash
python scripts/cluster_analysis.py
```
## Main Files
## Main Files
- `scripts/cluster_analysis.py` - Cluster analysis script
- `notebooks/cluster_analysis.ipynb` - Cluster analysis notebook

  
## Outputs
Running the cluster analysis script generates:
- `data/processed/customer_clustered_data.csv` - Segmented data
- `reports/cluster_analysis_summary.txt` - Cluster analysis summary
- `reports/cluster_profiles_summary.csv` - Cluster analysis summary data
- `outputs/plots/` - All visulizations

## Tools Used
- Python
- Pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook