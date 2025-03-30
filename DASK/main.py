import logging
from load_data import DatasetLoader
from data_processor import DataPreprocessor
from XGBoost import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_engines(loader):
    logging.info("Comparing performance of Pandas vs. Dask...")

    pd_clinical, pd_time1 = loader.load_with_pandas(dataset_type='clinical', file_format='excel')
    pd_gene, pd_time2 = loader.load_with_pandas(dataset_type='gene', delimiter='\t', file_format='csv', comment='!')

    dask_clinical, dask_time1 = loader.load_with_dask(dataset_type='clinical', file_format='excel')
    dask_gene, dask_time2 = loader.load_with_dask(dataset_type='gene', delimiter='\t', file_format='csv', comment='!')

    loader.print_performance(pd_time1, dask_time1,
                             *loader.compare_memory_usage(dask_clinical, pd_clinical),
                             dataset_name="Clinical Data")

    loader.print_performance(pd_time2, dask_time2,
                             *loader.compare_memory_usage(dask_gene, pd_gene),
                             dataset_name="Gene Expression Data")

def preprocess_and_engineer(data):
    processor = DataPreprocessor(data)
    processor.handle_missing_values()
    processor.encode_tumor_subtype()
    processor.filter_low_quality_genes(threshold=0.1)
    processor.normalize_gene_data()
    processor.perform_eda()
    return processor

def run_training(processor, test_ratio=0.2):
    X_train, X_test, y_train, y_test = processor.split_dataset(test_size=test_ratio)
    classifier = ModelTrainer(X_train, y_train, X_test, y_test)
    duration = classifier.train_model()
    results = classifier.evaluate_model()

    logging.info(f"\nTraining Time: {duration:.4f} seconds")
    for k, v in results.items():
        logging.info(f"{k}: {v:.4f}")

def pipeline():
    # Setup paths
    loader = DatasetLoader(
        gene_data_path="D:/Programming/Programming 6/DASK/data/GSE58661_series_matrix.txt.gz",
        clinical_data_path="D:/Programming/Programming 6/DASK/data/Lung3.metadata.xls"
    )

    # Optional performance benchmark
    compare_engines(loader)

    # Load and integrate datasets
    logging.info("\nPreparing merged dataset for analysis...")
    merged_data = loader.load_and_preprocess().compute()

    # Preprocessing
    processor = preprocess_and_engineer(merged_data)

    # Train + evaluate
    run_training(processor)

if __name__ == "__main__":
    pipeline()
