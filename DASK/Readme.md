#  Scalable Lung Cancer Classification with Dask and XGBoost

This project builds a **scalable machine learning pipeline** for classifying lung cancer subtypes (e.g., Squamous vs. Others) using large-scale **gene expression data** and **clinical metadata**. The pipeline leverages **Dask** for distributed data processing and **XGBoost** for classification, offering speed and memory advantages over traditional pandas/scikit-learn workflows.


##  Dataset
- **Gene Expression**: `GSE58661_series_matrix.txt.gz` (GEO)
- **Clinical Metadata**: `Lung3.metadata.xls` (TCIA)

##  Project Structure

├── main.py                 # Pipeline entry point
├── load_data.py            # Dask + pandas data loaders and merger
├── data_processor.py       # Preprocessing, encoding, filtering, EDA
├── XGBoost.py              # Model training and evaluation
├── README.md               # Project overview and usage guide
├── Documentation           # Project explanations


## How to Run
1.
Install the dependencies:
```bash
pip install dask[complete] xgboost scikit-learn pandas matplotlib seaborn openpyxl
```

2.  Update Paths
Edit `main.py` to set the paths for your datasets:
```python
gene_data_path = 'path/to/GSE58661_series_matrix.txt.gz'
clinical_data_path = 'path/to/Lung3.metadata.xls'
```

3.  Execute the Pipeline
```bash
python main.py
```

## Results
| Metric      | Value     |
|-------------|-----------|
| Accuracy    | 94.44%    |
| Precision   | 88.89%    |
| Recall      | 100.00%   |
| F1 Score    | 94.12%    |
| AUC-ROC     | 95.00%    |

Author: Maryam Esmaeili
