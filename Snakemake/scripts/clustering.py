import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

class ClusteringAndOutlierDetection:
    @staticmethod
    def apply_hac(features, n_clusters=None, distance_threshold=0.5):
        """Apply Hierarchical Agglomerative Clustering (HAC) and return labels."""
        
        # Initialize HAC model
        hac = AgglomerativeClustering(
            n_clusters=n_clusters, 
            distance_threshold=distance_threshold, 
            linkage='average'
        )

        # Fit model and obtain cluster labels
        labels = hac.fit_predict(features)

        return labels  # Ensure labels are returned properly


    @staticmethod
    def detect_outliers_isolation(features, contamination=0.05):
        """Detect outliers using Isolation Forest."""
        model = IsolationForest(contamination=contamination, random_state=42)
        return model.fit_predict(features)

    @staticmethod
    def detect_outliers_elliptic(features, contamination=0.05):
        """Detect outliers using Elliptic Envelope."""
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        return model.fit_predict(features)

    @staticmethod
    def compare_outliers(df, outlier_isolation, outlier_elliptic):
        """
        Compare outlier results from Isolation Forest and Elliptic Envelope.
        Label outliers detected by one or both methods.
        """
        combined_status = []
        for iso, ell in zip(outlier_isolation, outlier_elliptic):
            if iso == -1 and ell == -1:
                combined_status.append("Both")  # Detected by both methods
            elif iso == -1:
                combined_status.append("Isolation Only")  # Only Isolation Forest detected
            elif ell == -1:
                combined_status.append("Elliptic Only")  # Only Elliptic Envelope detected
            else:
                combined_status.append("Inlier")  # Normal point

        df["combined_outlier"] = combined_status
        print("Final Outlier Label Counts:")
        print(df["combined_outlier"].value_counts())

        return df
