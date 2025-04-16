import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import necessary modules
from evaluation import print_and_save_validation_scores
from load_data import DataLoader
from clean import DataCleaner
from features import FeatureProcessor
from clustering import ClusteringAndOutlierDetection
from cluster_data import CombinedDatasetProcessor
from visualisation import Visualizer
from cluster_analysis import MixedClusterIdentifier
from substitute_analysis import NaturalSubstituteSaver, SubstituteAnalyzer
from molecular_pipeline import MolecularPipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Define directories for outputs
    DATA_DIR = os.path.join(os.getcwd(), "data")  # CSV storage
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")  # Reports and images
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use a single report file (report.txt) in the output folder
    report_file = os.path.join(OUTPUT_DIR, "report.txt")

    # Load the merged features file
    FEATURES_FILE = os.path.join(DATA_DIR, "featurized", "merged_features.csv")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Input file not found: {FEATURES_FILE}")

    # Load the dataset
    combined_df = pd.read_csv(FEATURES_FILE, sep=",")
    logging.info(f"Loaded dataset shape: {combined_df.shape}")

    # Extract numerical features for outlier detection
    numerical_features = combined_df.select_dtypes(include=[np.number])
    logging.info("Detecting outliers using Isolation Forest and Elliptic Envelope...")

    # Apply outlier detection
    isolation_outliers = ClusteringAndOutlierDetection.detect_outliers_isolation(numerical_features)
    elliptic_outliers = ClusteringAndOutlierDetection.detect_outliers_elliptic(numerical_features)

    # Combine outlier results
    combined_df = ClusteringAndOutlierDetection.compare_outliers(combined_df, isolation_outliers, elliptic_outliers)

    # Plot and save outlier results
    outlier_plot_path = os.path.join(OUTPUT_DIR, "outlier_detection.png")
    Visualizer.plot_outliers(
        numerical_features,  # Use raw numerical features before clustering
        combined_df["combined_outlier"].values,
        "Outlier Detection",
        output_path=outlier_plot_path
    )

    # Run the molecular pipeline (including HAC clustering)
    molecular_pipeline = MolecularPipeline(combined_df, "Combined Dataset")
    clustered_df, hac_labels, reduced_features = molecular_pipeline.process_and_cluster()

    # Save clustered data (only required columns)
    processed_data_path = os.path.join(DATA_DIR, "processed_data.csv")
    essential_columns = ["canonical_smiles", "source", "hac_cluster"]
    clustered_df[essential_columns].to_csv(processed_data_path, index=False)
    logging.info(f"Processed data saved to {processed_data_path}.")

    # Save clustered dataset
    clustered_output_path = os.path.join(DATA_DIR, "clustered_data.csv")
    clustered_df.to_csv(clustered_output_path, index=False)
    logging.info(f"Clustered data saved to {clustered_output_path}.")

    # Compute and append HAC validation scores
    silhouette_hac, davies_bouldin_hac, calinski_harabasz_hac = print_and_save_validation_scores(
        reduced_features, hac_labels
    )

    # Further process the combined dataset (e.g., mixed clusters)
    combined_df, combined_hac_labels, combined_reduced_features = CombinedDatasetProcessor.process_combined_dataset(
        clustered_df, n_neighbors=5, min_dist=0.01
    )

    # Compute and append mixed cluster validation scores
    silhouette_mixed, davies_bouldin_mixed, calinski_harabasz_mixed = print_and_save_validation_scores(
        combined_reduced_features, combined_hac_labels
    )

    # Define the output path for the cluster visualization
    mixed_cluster_plot_path = os.path.join(OUTPUT_DIR, "mixed_clusters.png")

    # Identify mixed clusters first
    mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, "hac_cluster")

    # Ensure only mixed cluster data is used
    mixed_df = combined_df[combined_df["hac_cluster"].isin(mixed_clusters)].copy().reset_index(drop=True)

    # Convert `combined_reduced_features` into a DataFrame without relying on index alignment
    combined_reduced_features_df = pd.DataFrame(combined_reduced_features).reset_index(drop=True)

    # Convert `reduced_features` to a DataFrame to allow proper indexing
    reduced_features_df = pd.DataFrame(reduced_features).reset_index(drop=True)

    # Extract the correctly indexed mixed cluster features using `.iloc[]`
    mixed_reduced_features = reduced_features_df.iloc[mixed_df.index].to_numpy()


    # Pass corrected data to visualization
    MixedClusterIdentifier.find_and_visualize_mixed_clusters(
        mixed_df, 
        mixed_reduced_features,  
        cluster_column="hac_cluster", 
        title="Mixed Clusters (HAC)", 
        output_path=mixed_cluster_plot_path
    )

    # Plot Tanimoto similarity histogram
    tanimoto_output_path = os.path.join(OUTPUT_DIR, "tanimoto.png")
    df_synthetic = combined_df[combined_df["source"] == "Synthetic"]
    df_natural = combined_df[combined_df["source"] == "Natural"]
    Visualizer.plot_tanimoto_histogram(df_synthetic, df_natural, output_path=tanimoto_output_path)

    # Save natural substitutes in mixed clusters
    natural_substitutes_file = os.path.join(DATA_DIR, "natural_substitutes.csv")
    NaturalSubstituteSaver.save_natural_substitutes_to_csv(
        combined_df, cluster_column="hac_cluster", source_column="source", output_file=natural_substitutes_file
    )

    # Identify common natural substitutes
    substitutes_df = SubstituteAnalyzer.find_common_natural_substitutes(combined_df, cluster_column="hac_cluster", similarity_threshold=0.8)

    # Plot property differences if substitutes exist
    if not substitutes_df.empty:
        SubstituteAnalyzer.plot_property_differences(substitutes_df, output_dir=OUTPUT_DIR)
    else:
        logging.info("No substitutes found. Plot will not be generated.")
    
    # Get top unique natural substitutes as DataFrame
    top_natural_substitutes = SubstituteAnalyzer.find_top_unique_natural_substitutes_with_visuals("output/substitutes_report.csv")


    # Convert DataFrame to a formatted string and write to file
    with open(report_file, "a") as f:
        f.write("\nHAC Clustering Validation Scores:\n")
        f.write(f"  Silhouette Score: {silhouette_hac:.4f}\n")
        f.write(f"  Davies-Bouldin Index: {davies_bouldin_hac:.4f}\n")
        f.write(f"  Calinski-Harabasz Index: {calinski_harabasz_hac:.4f}\n")

        f.write("\nMixed Cluster Validation Scores:\n")
        f.write(f"  Silhouette Score: {silhouette_mixed:.4f}\n")
        f.write(f"  Davies-Bouldin Index: {davies_bouldin_mixed:.4f}\n")
        f.write(f"  Calinski-Harabasz Index: {calinski_harabasz_mixed:.4f}\n")

        f.write("\nTop Unique Natural Substitutes:\n")
        f.write(top_natural_substitutes.to_string(index=False))  # Convert DataFrame to string
        f.write("\n")


if __name__ == "__main__":
    main()
