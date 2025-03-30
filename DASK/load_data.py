import dask.dataframe as dd
import pandas as pd
import dask
import time

class DatasetLoader:
    def __init__(self, gene_data_path, clinical_data_path):
        self.gene_data_path = gene_data_path
        self.clinical_data_path = clinical_data_path

    def load_with_pandas(self, dataset_type='gene', delimiter=None, file_format='csv', **kwargs):
        start_time = time.time()
        if dataset_type == 'gene':
            file_path = self.gene_data_path
        elif dataset_type == 'clinical':
            file_path = self.clinical_data_path
        else:
            raise ValueError("Invalid dataset_type. Use 'gene' or 'clinical'.")

        if file_format == 'csv':
            df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        elif file_format == 'excel':
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'excel'.")

        loading_time = time.time() - start_time
        return df, loading_time

    def load_with_dask(self, dataset_type='gene', delimiter=None, file_format='csv', npartitions=4, **kwargs):
        start_time = time.time()
        if dataset_type == 'gene':
            file_path = self.gene_data_path
        elif dataset_type == 'clinical':
            file_path = self.clinical_data_path
        else:
            raise ValueError("Invalid dataset_type. Use 'gene' or 'clinical'.")

        if file_format == 'csv':
            df = dd.read_csv(file_path, delimiter=delimiter, **kwargs)
        elif file_format == 'excel':
            pandas_df = pd.read_excel(file_path, **kwargs)
            df = dd.from_pandas(pandas_df, npartitions=npartitions)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'excel'.")

        loading_time = time.time() - start_time
        return df, loading_time

    def compare_memory_usage(self, dask_df, pandas_df):
        dask_memory = dask_df.memory_usage().compute().sum() / (1024 ** 2)
        pandas_memory = pandas_df.memory_usage().sum() / (1024 ** 2)
        return dask_memory, pandas_memory

    def print_performance(self, pandas_load_time, dask_load_time, pandas_mem, dask_mem, dataset_name):
        print(f"\n--- Performance Metrics for {dataset_name} ---")
        print(f"Time taken by Pandas: {pandas_load_time:.2f} seconds")
        print(f"Time taken by Dask: {dask_load_time:.2f} seconds")
        print(f"Pandas Memory Usage: {pandas_mem:.2f} MB")
        print(f"Dask Memory Usage: {dask_mem:.2f} MB")

    def load_and_preprocess(self):
        clinical_ddf, _ = self.load_with_dask(dataset_type='clinical', file_format='excel')
        gene_ddf, _ = self.load_with_dask(
            dataset_type='gene',
            delimiter='\t',
            file_format='csv',
            comment='!',  # Skip metadata rows starting with !
            blocksize=None  # Required for gzip
            )
        print("\nGene Expression Head:")
        print(gene_ddf.head())
        
        clinical_df = clinical_ddf.compute()
        gene_df = gene_ddf.compute()

        return delayed_merge(clinical_df, gene_df)

@dask.delayed
def delayed_merge(clinical_df, gene_df):
    if "ID_REF" not in gene_df.columns:
        raise KeyError("Expected column 'ID_REF' not found in gene expression data")

    print("\nTransposing Gene Expression Data...")
    gene_df = gene_df.set_index("ID_REF").T.reset_index()
    gene_df.rename(columns={"index": "GSM_ID"}, inplace=True)

    if len(gene_df) != len(clinical_df):
        raise ValueError("Mismatch in number of samples between metadata and gene expression data.")

    clinical_df = clinical_df.sort_values(by="sample.name").reset_index(drop=True)
    gene_df = gene_df.sort_values(by="GSM_ID").reset_index(drop=True)

    mapping = dict(zip(gene_df["GSM_ID"], clinical_df["sample.name"]))
    gene_df["sample.name"] = gene_df["GSM_ID"].map(mapping)

    print("\nMerging Clinical and Gene Data...")
    merged = pd.merge(clinical_df, gene_df.drop(columns=["GSM_ID"]), on="sample.name", how="inner")
    print("Merged Data Shape:", merged.shape)
    return merged

