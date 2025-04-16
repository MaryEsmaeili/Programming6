#!/usr/bin/env python3
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def print_and_save_validation_scores(features, labels):
    """
    Compute and print clustering validation scores, then save them to a text file.
    
    Parameters:
      features: array-like, e.g. UMAP-reduced features.
      labels: array-like, clustering labels.
      
    Returns:
      A tuple (silhouette, davies_bouldin, calinski_harabasz)
    """
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    
    report = (
        "Validation Scores for Combined Dataset HAC:\n"
        f"  Silhouette Score: {silhouette:.4f}\n"
        f"  Davies-Bouldin Index: {davies_bouldin:.4f}\n"
        f"  Calinski-Harabasz Index: {calinski_harabasz:.4f}\n"
    )


    return silhouette, davies_bouldin, calinski_harabasz
