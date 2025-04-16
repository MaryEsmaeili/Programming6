from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

class Visualizer:
    
    @staticmethod
    def visualize_clusters(data, labels, title, output_path=None):
        """Visualize clustering results in UMAP-reduced space.
           If output_path is provided, save the plot as a PNG file.
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Cluster plot saved to {output_path}")
        else:
            plt.show()

    @staticmethod
    def plot_outliers(features, outlier_labels, title, output_path="output/outlier_detection.png"):
        """Plot outliers detected by Isolation Forest and Elliptic Envelope."""
        print("Unique outlier labels detected:", np.unique(outlier_labels))
        print("Outlier Label Counts:", pd.Series(outlier_labels).value_counts())

        # Apply PCA for better visualization if UMAP is collapsing data
        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

        plt.figure(figsize=(10, 8))
        
        colors = {"Inlier": "blue", "Both": "red", "Isolation Only": "orange", "Elliptic Only": "purple"}
        
        unique_labels = set(outlier_labels)
        print("Final unique outlier labels before plotting:", unique_labels)
        
        # Plot outliers first so they remain visible
        for label in ["Both", "Isolation Only", "Elliptic Only"]:
            mask = np.array(outlier_labels) == label
            if mask.sum() > 0:
                plt.scatter(features[mask, 0], features[mask, 1], 
                            c=colors[label], label=label, 
                            alpha=0.9, s=200, edgecolors='black')
        
        # Now plot inliers on top to avoid covering outliers
        mask_inliers = np.array(outlier_labels) == "Inlier"
        plt.scatter(features[mask_inliers, 0], features[mask_inliers, 1], 
                    c="blue", label="Inlier", alpha=0.5, s=100, edgecolors='black')
        
        plt.legend(title="Outlier Type")
        plt.title(title)
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        
        # Save the figure explicitly
        plt.savefig(output_path)
        plt.show()
        print(f"Outlier plot saved to {output_path}")

    @staticmethod
    def visualize_synthetic_natural_common(data, sources, title, output_path=None):
        """Visualize synthetic, natural, and common compounds in UMAP-reduced space.
        If output_path is provided, save the plot as a PNG file.
        """
        plt.figure(figsize=(12, 10))
        plt.clf() 

        color_map = {"Synthetic": "blue", "Natural": "green", "Common": "purple"}

        # Ensure `data` and `sources` are the same length before visualization
        if len(data) != len(sources):
            raise ValueError(f"Data length ({len(data)}) and sources length ({len(sources)}) do not match! Check indexing.")


        # Use the correct clustering dataset
        data = np.array(data)  # Ensure it's a NumPy array


        for source in ["Synthetic", "Natural", "Common"]:
            mask = (np.array(sources) == source)
            if mask.sum() > 0:
                plt.scatter(data[mask, 0], data[mask, 1], 
                            c=color_map[source], label=source,
                            alpha=0.5, s=50)
        
        plt.legend(title="Compound Types", loc='upper right')
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        
        if output_path:
            plt.savefig(output_path)  #Ensure correct file saving
            plt.close()
            print(f"Synthetic/Natural common plot saved to {output_path}")
        else:
            plt.show()


    @staticmethod
    def plot_tanimoto_histogram(df_synthetic, df_natural, output_path="output/tanimoto.png"):
        """Plot a histogram comparing average Tanimoto distances for synthetic and natural compounds.
           If output_path is provided, save the plot as a PNG file.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(df_synthetic['avg_tanimoto'], bins=30, alpha=0.7, label='Synthetic', color='blue')
        plt.hist(df_natural['avg_tanimoto'], bins=30, alpha=0.7, label='Natural', color='green')
        plt.xlabel("Average Tanimoto Distance")
        plt.ylabel("Frequency")
        plt.title("Comparison of Average Tanimoto Distances")
        plt.legend()
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Tanimoto histogram saved to {output_path}")
        else:
            plt.show()
