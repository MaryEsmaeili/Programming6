import pandas as pd
from descriptor_fingerprint_processor import DescriptorFingerprintProcessor
from clustering import ClusteringAndOutlierDetection
from features import FeatureProcessor
import os

class MolecularPipeline:
    def __init__(self, combined_df, dataset_name, n_neighbors=5, min_dist=0.01):
        self.combined_df = combined_df
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def process_and_cluster(self):
        """Full pipeline to process, detect outliers, and cluster using HAC."""

        # Ensure df is properly assigned
        df = self.combined_df.copy()

        print(f"{self.dataset_name} initial data shape: {df.shape}")

        # Ensure canonical_smiles exists before computing fingerprints
        if "canonical_smiles" not in df.columns:
            raise KeyError("Missing 'canonical_smiles' column in input dataset.")

        # Compute descriptors
        if "MolWt" not in df.columns:
            descriptors = df['canonical_smiles'].apply(DescriptorFingerprintProcessor.compute_descriptors)
            df = pd.concat([df, pd.DataFrame(descriptors.tolist())], axis=1)


        # Generate fingerprints
        df['fingerprints'] = DescriptorFingerprintProcessor.generate_rdkit_fingerprints(df['canonical_smiles'])

        # **Confirm fingerprints exist before proceeding**
        if df['fingerprints'].isna().all():
            raise ValueError("All fingerprints are NaN. Check input SMILES strings.")

        # Compute average Tanimoto similarity
        df['avg_tanimoto'] = DescriptorFingerprintProcessor.compute_avg_tanimoto_in_batches(
            fingerprints=df['fingerprints'].tolist(), batch_size=5000
        )

        # **Detect Outliers Before Clustering**
        numerical_features = FeatureProcessor.prepare_features(df)

        print("\ Detecting outliers using Isolation Forest and Elliptic Envelope...")
        isolation_outliers = ClusteringAndOutlierDetection.detect_outliers_isolation(numerical_features)
        elliptic_outliers = ClusteringAndOutlierDetection.detect_outliers_elliptic(numerical_features)

        # Compare and mark outliers
        df = ClusteringAndOutlierDetection.compare_outliers(df, isolation_outliers, elliptic_outliers)

        # Remove outliers before clustering
        # Ensure only inliers are used
        df_filtered = df[df["combined_outlier"] == "Inlier"].copy()
        if df_filtered.empty:
            raise ValueError("No valid inliers remain after outlier filtering. Adjust contamination levels.")


        # Reduce dimensions with UMAP
        reduced_features = FeatureProcessor.apply_umap(
            FeatureProcessor.prepare_features(df_filtered), 
            n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist
        )
        
        # **Apply HAC Clustering**
        hac_labels = ClusteringAndOutlierDetection.apply_hac(reduced_features, n_clusters=None, distance_threshold=1.5)
        df_filtered['hac_cluster'] = hac_labels

        # **Confirm HAC Cluster Labels Exist**
        print("\nHAC Clustering applied. Checking available columns:")
        print(df_filtered.columns)

        return df_filtered, hac_labels, reduced_features
