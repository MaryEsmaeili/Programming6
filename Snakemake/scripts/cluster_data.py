from features import FeatureProcessor
from visualisation import Visualizer
from clustering import ClusteringAndOutlierDetection
from evaluation import print_and_save_validation_scores

class CombinedDatasetProcessor:
    @staticmethod
    def process_combined_dataset(combined_df, n_neighbors=5, min_dist=0.01):
        """Apply HAC clustering on the combined dataset."""

        print(f"Combined dataset shape: {combined_df.shape}")

        # Ensure only inliers are used before clustering
        filtered_df = combined_df[combined_df["combined_outlier"] == "Inlier"].copy()
        if filtered_df.empty:
            raise ValueError("No valid inliers remain after outlier filtering. Adjust contamination levels.")

        # Prepare features and fingerprints
        fingerprints = filtered_df['fingerprints'].tolist()
        numerical_features = FeatureProcessor.prepare_features(filtered_df)

        # Reduce dimensions with UMAP only on inliers
        reduced_features = FeatureProcessor.apply_umap(numerical_features, n_neighbors=n_neighbors, min_dist=min_dist)

        # Apply HAC on UMAP-reduced features
        hac_labels = ClusteringAndOutlierDetection.apply_hac(reduced_features, n_clusters=None, distance_threshold=1.5)
        filtered_df['hac_cluster'] = hac_labels

        # Visualize the final clustering without outliers
        Visualizer.visualize_clusters(
            reduced_features,
            hac_labels,
            "Combined Dataset Clusters (HAC)",
            output_path="output/combined_clusters.png"
        )

        return filtered_df, hac_labels, reduced_features
