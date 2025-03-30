import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dask

class DataPreprocessor:
    def __init__(self, merged_df):
        self.df = merged_df.copy()

    def handle_missing_values(self):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns

        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        if len(categorical_cols) > 0:
            mode_vals = self.df[categorical_cols].mode().iloc[0]
            self.df[categorical_cols] = self.df[categorical_cols].fillna(mode_vals)
        return self.df

    def filter_low_quality_genes(self, threshold=0.1):
        numeric_cols = self.df.select_dtypes(include="number").columns.difference(["Encoded_Subtype"])
        variances = self.df[numeric_cols].var()
        high_var_genes = variances[variances > threshold].index
        print(f"Retaining {len(high_var_genes)} high-variance features")
        self.df = self.df[high_var_genes.tolist() + ["Encoded_Subtype"]]
        return self.df

    def normalize_gene_data(self):
        numeric_cols = self.df.select_dtypes(include="number").columns.difference(["Encoded_Subtype"])
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df

    def encode_tumor_subtype(self):
        print("\nEncoding Tumor Subtype...")
        possible_cols = [
            "characteristics.tag.histology",  # <--- correct column based on your screenshot
            "characteristics_tag_histology", "histology", "tumor_subtype",
            "subtype", "characteristics_tag_tumor_type"
        ]
        for col in possible_cols:
            if col in self.df.columns:
                self.df["TumorSubtype"] = self.df[col].astype(str).str.lower()
                self.df["Encoded_Subtype"] = self.df["TumorSubtype"].apply(lambda x: 1 if "squamous" in x else 0)
                return self.df

        print("Tumor subtype column not found. Skipping encoding.")
        self.df["Encoded_Subtype"] = -1
        return self.df

    def perform_eda(self):
        try:
            @dask.delayed
            def compute_summary(df):
                print("\nClinical Data Summary:")
                return df.describe()

            @dask.delayed
            def compute_variance(df):
                numeric_cols = df.select_dtypes(include="number").columns
                return df[numeric_cols].var().sort_values(ascending=False)

            summary = compute_summary(self.df)
            variances = compute_variance(self.df)

            summary_result, var_result = dask.compute(summary, variances)
            print(summary_result)

            print("\nTop 10 Most Variable Features:")
            print(var_result.head(10))

            if "Encoded_Subtype" in self.df.columns:
                subtype_counts = self.df["Encoded_Subtype"].value_counts()
                plt.figure(figsize=(6, 4))
                sns.barplot(x=subtype_counts.index, y=subtype_counts.values)
                plt.title("Tumor Subtype Distribution")
                plt.xlabel("Subtype")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"EDA failed: {e}")

    def split_dataset(self, test_size=0.2, random_state=42):
        valid = self.df[self.df["Encoded_Subtype"] != -1]
        X = valid.drop(columns=["Encoded_Subtype", "TumorSubtype"], errors="ignore")
        y = valid["Encoded_Subtype"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)