import pandas as pd
import numpy as np
from rdkit import DataStructs
import logging
import os
import matplotlib.pyplot as plt
from cluster_analysis import MixedClusterIdentifier
from rdkit import Chem
from rdkit.Chem import Draw
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from tabulate import tabulate

class NaturalSubstituteSaver:
    @staticmethod
    def save_natural_substitutes_to_csv(combined_df, cluster_column, source_column, output_file):
        mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, cluster_column)
        natural_substitutes = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df[source_column] == "Natural")
        ]
        natural_substitutes.to_csv(output_file, index=False)
        logging.info(f"Natural substitutes saved to {output_file}")

class MoleculeVisualizer:
    """
    A class for visualizing a single molecule from a SMILES string 
    """
    def __init__(self, smiles):
        """
        Initializes the MoleculeVisualizer with a single SMILES string and converts it
        to an RDKit molecule object.
        """
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles) if smiles else None
    
    def draw(self, save_path=None):
        """
        Draws the molecule and saves or returns the image.
        """
        if self.mol is None:
            logging.warning(f"Invalid SMILES: {self.smiles}")
            return None
        
        img = Draw.MolToImage(self.mol)
        if save_path:
            img.save(save_path)
        else:
            return img

class SubstituteAnalyzer:
    @staticmethod
    def find_top_unique_natural_substitutes_with_visuals(substitutes_file, output_dir="output/molecule_images"):
        """Finds top unique natural substitutes and generates molecular images for visualization."""
        try:
            substitutes_df = pd.read_csv(substitutes_file)
            if "Tanimoto_Similarity" not in substitutes_df.columns:
                raise KeyError("Tanimoto_Similarity column is missing in the substitutes report.")
            
            os.makedirs(output_dir, exist_ok=True)
            
            top_natural_substitutes = substitutes_df.loc[
                substitutes_df.groupby("Natural_Compound")["Tanimoto_Similarity"].idxmax()
            ]
            
            report_data = []  # Store table data for report

            # Generate molecule images for both Natural and Synthetic compounds
            for _, row in top_natural_substitutes.iterrows():
                natural_image_path = f"output/molecule_images/Natural_{row.name}.png"
                synthetic_image_path = f"output/molecule_images/Synthetic_{row.name}.png"
                
                MoleculeVisualizer(row["Natural_Compound"]).draw(save_path=os.path.join(output_dir, f"Natural_{row.name}.png"))
                MoleculeVisualizer(row["Synthetic_Compound"]).draw(save_path=os.path.join(output_dir, f"Synthetic_{row.name}.png"))
                
                # Add data to report
                report_data.append([
                f"![Natural]({natural_image_path})",
                f"![Synthetic]({synthetic_image_path})",
                row["Tanimoto_Similarity"],
                row.get("MolWt_Diff", 0),
                row.get("LogP_Diff", 0),
                row.get("TPSA_Diff", 0),
                row.get("HDonors_Diff", 0),
                row.get("HAcceptors_Diff", 0)
            ])

            
            report_table = tabulate(report_data, headers=["Natural Compound", "Synthetic Compound", "Tanimoto Similarity", "MolWt Diff", "LogP Diff", "TPSA Diff", "HDonors Diff", "HAcceptors Diff"], tablefmt="github")
            
            with open("output/report.txt", "a") as f:
                f.write("\nTop Unique Natural Substitutes:\n")
                f.write(report_table)
                f.write("\n")
            
            print("Top Unique Natural Substitutes with visualizations added to report.")
            return top_natural_substitutes
        except FileNotFoundError:
            logging.error(f"File {substitutes_file} not found. Ensure that the substitutes report has been generated.")
        except Exception as e:
            logging.error(f"Error processing the substitutes report: {e}")

    @staticmethod
    def find_common_natural_substitutes(combined_df, cluster_column, similarity_threshold=0.8):
        mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, cluster_column)

        natural_in_mixed = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df["source"] == "Natural")
        ]
        synthetic_in_mixed = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df["source"] == "Synthetic")
        ]

        logging.info(f"Natural compounds in mixed clusters: {len(natural_in_mixed)}")
        logging.info(f"Synthetic compounds in mixed clusters: {len(synthetic_in_mixed)}")

        substitutes = []
        for cluster in mixed_clusters:
            natural_cluster = combined_df[
                (combined_df[cluster_column] == cluster) & (combined_df["source"] == "Natural")
            ]
            synthetic_cluster = combined_df[
                (combined_df[cluster_column] == cluster) & (combined_df["source"] == "Synthetic")
            ]

            for _, nat_row in natural_cluster.iterrows():
                for _, syn_row in synthetic_cluster.iterrows():
                    nat_fp = nat_row["fingerprints"]
                    syn_fp = syn_row["fingerprints"]

                    if nat_fp is None or syn_fp is None:
                        continue  # Skip missing fingerprints

                    tanimoto_sim = DataStructs.TanimotoSimilarity(nat_fp, syn_fp)
                    if tanimoto_sim >= similarity_threshold:
                        property_diff = {
                            "MolWt_Diff": abs(float(nat_row.get("MolWt", 0)) - float(syn_row.get("MolWt", 0))),
                            "LogP_Diff": abs(float(nat_row.get("LogP", 0)) - float(syn_row.get("LogP", 0))),
                            "TPSA_Diff": abs(float(nat_row.get("TPSA", 0)) - float(syn_row.get("TPSA", 0))),
                            "HDonors_Diff": abs(int(nat_row.get("NumHDonors", 0)) - int(syn_row.get("NumHDonors", 0))),
                            "HAcceptors_Diff": abs(int(nat_row.get("NumHAcceptors", 0)) - int(syn_row.get("NumHAcceptors", 0))),
                        }
                        substitutes.append({
                            "Natural_Compound": nat_row["canonical_smiles"],
                            "Synthetic_Compound": syn_row["canonical_smiles"],
                            "Cluster": cluster,
                            "Tanimoto_Similarity": tanimoto_sim,
                            **property_diff
                        })

        substitutes_df = pd.DataFrame(substitutes)

        # Ensure required columns exist
        required_columns = ["Natural_Compound", "Synthetic_Compound", "Cluster", "Tanimoto_Similarity",
                            "MolWt_Diff", "LogP_Diff", "TPSA_Diff", "HDonors_Diff", "HAcceptors_Diff"]

        for col in required_columns:
            if col not in substitutes_df.columns:
                substitutes_df[col] = 0  # Fill missing columns with default values

        substitutes_df.to_csv("output/substitutes_report.csv", index=False)
        logging.info("Report saved to output/substitutes_report.csv")

        return substitutes_df
    
    @staticmethod
    def plot_property_differences(substitutes_df, output_dir="output"):
        properties = ["MolWt_Diff", "LogP_Diff", "TPSA_Diff", "HDonors_Diff", "HAcceptors_Diff"]
        plt.figure(figsize=(15, 10))

        for i, prop in enumerate(properties, start=1):
            plt.subplot(2, 3, i)
            substitutes_df[prop].astype(float).hist(bins=20, alpha=0.7, color="blue", edgecolor="black")
            plt.title(f"Distribution of {prop}")
            plt.xlabel(prop)
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.suptitle("Differences in Molecular Properties (Natural vs Synthetic)", y=1.02, fontsize=16)

        # Save plot
        output_path = os.path.join(output_dir, "property_differences.png")
        plt.savefig(output_path)
        plt.show()

        logging.info(f"Plot saved to {output_path}")
