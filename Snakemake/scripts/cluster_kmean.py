import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import umap
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class KMeansTanimotoClustering:
    def __init__(self, smiles_list, numeric_features, output_dir="output/"):
        self.smiles_list = smiles_list
        self.numeric_features = numeric_features
        self.output_dir = output_dir

    def compute_tanimoto_distance(self):
        """Computes the Tanimoto distance matrix from molecular fingerprints."""
        fingerprints = []
        for smiles in self.smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fingerprints.append(fp)
            else:
                fingerprints.append(None)

        n = len(fingerprints)
        tanimoto_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if fingerprints[i] is not None and fingerprints[j] is not None:
                    tanimoto_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                else:
                    tanimoto_matrix[i, j] = 0  # Assign lowest similarity if invalid SMILES

        tanimoto_distance = 1 - tanimoto_matrix  # Convert similarity to distance
        return tanimoto_distance

    def apply_kmeans(self, data, n_clusters):
        """Applies KMeans clustering and returns labels along with evaluation metrics."""
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        # Compute evaluation scores
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)

        return labels, silhouette, davies_bouldin, calinski_harabasz

    def visualize_clusters(self, data, labels, n_clusters):
        """Uses UMAP for dimensionality reduction and visualizes KMeans clusters."""
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, metric="precomputed", random_state=42)
        reduced_data = reducer.fit_transform(data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="tab10", alpha=0.8)
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title(f"KMeans Clustering with Tanimoto Distance (n_clusters={n_clusters})")

        plot_path = f"{self.output_dir}kmeans_tanimoto_{n_clusters}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return plot_path

    def run_all(self):
        """Runs KMeans for multiple cluster sizes and saves results."""
        tanimoto_distance = self.compute_tanimoto_distance()

        results = []
        for n_clusters in range(4, 11):  # Clusters from 4 to 10
            labels, silhouette, davies_bouldin, calinski_harabasz = self.apply_kmeans(tanimoto_distance, n_clusters)
            plot_path = self.visualize_clusters(tanimoto_distance, labels, n_clusters)

            # Save results to list
            results.append(f"Clusters: {n_clusters}\n"
                           f"  Silhouette Score: {silhouette:.4f}\n"
                           f"  Davies-Bouldin Index: {davies_bouldin:.4f}\n"
                           f"  Calinski-Harabasz Index: {calinski_harabasz:.4f}\n"
                           f"  Plot saved at: {plot_path}\n")

        # Write results to file
        with open(f"{self.output_dir}kmean_tanimoto.txt", "w") as f:
            f.writelines(results)
        print(" KMeans clustering results with Tanimoto distance saved to kmean_tanimoto.txt.")

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]  # Expecting preprocessed dataset path
    df = pd.read_csv(input_file)

    # Ensure KMeans runs AFTER outliers are removed
    if "combined_outlier" in df.columns:
        print("Filtering outliers before clustering...")
        df = df[df["combined_outlier"] == "Inlier"].reset_index(drop=True)
        print(f"Remaining dataset shape after removing outliers: {df.shape}")

    smiles_list = df["canonical_smiles"].tolist()
    numeric_features = df.select_dtypes(include=[np.number])

    kmeans_cluster = KMeansTanimotoClustering(smiles_list, numeric_features)
    kmeans_cluster.run_all()
