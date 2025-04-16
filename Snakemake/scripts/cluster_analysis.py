import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from visualisation import Visualizer
from clustering import ClusteringAndOutlierDetection
from features import FeatureProcessor

class MixedClusterIdentifier:
    @staticmethod
    def identify_mixed_clusters(combined_df, cluster_column):
        """Identify clusters containing both synthetic and natural compounds."""
        combined_df = combined_df[combined_df["combined_outlier"] == "Inlier"].copy()
        grouped = combined_df.groupby(cluster_column)
        mixed_clusters = []

        for cluster, group in grouped:
            sources = group['source'].unique()
            if len(sources) > 1:  # Clusters containing both Synthetic AND Natural
                mixed_clusters.append(cluster)

        return mixed_clusters

    @staticmethod
    def find_and_visualize_mixed_clusters(combined_df, reduced_features, cluster_column, title="Mixed Clusters", output_path=None):
        """Identify and visualize clusters containing both synthetic and natural compounds."""
        mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, cluster_column)
        mixed_df = combined_df[combined_df[cluster_column].isin(mixed_clusters) & (combined_df["combined_outlier"] == "Inlier")].copy()

        reduced_features_df = pd.DataFrame(reduced_features).reset_index(drop=True)
        mixed_df = mixed_df.reset_index(drop=True)
        mixed_reduced_features = reduced_features_df.loc[mixed_df.index].to_numpy()
        
        # Define color map for synthetic and natural compounds
        color_map = {"Synthetic": "red", "Natural": "blue"}
        source_colors = [color_map[source] for source in mixed_df['source']]
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(mixed_reduced_features[:, 0], mixed_reduced_features[:, 1], c=source_colors, alpha=0.7, s=50)
        
        legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[source], markersize=10) for source in color_map]
        plt.legend(legend_labels, color_map.keys(), title="Compound Type", loc='upper right')
        
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"Mixed Cluster plot saved to {output_path}")
        else:
            plt.show()
