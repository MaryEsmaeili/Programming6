import os
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from rdkit.Chem import AllChem, DataStructs, Descriptors

class FeatureProcessor:
    @staticmethod
    def prepare_features(df):
        """Prepare and scale features for clustering."""
        features = df[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']].fillna(0).values
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    @staticmethod
    def apply_umap(features, n_neighbors=15, min_dist=0.01, n_components=2, random_state=42):
        """Apply UMAP for dimensionality reduction."""
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state
        )
        return reducer.fit_transform(features)

    @staticmethod
    def compute_descriptors(smiles):
        """Compute molecular descriptors for a given SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            }
        return {"MolWt": None, "LogP": None, "TPSA": None, "NumHDonors": None, "NumHAcceptors": None}


def process_and_save_features(input_file, output_file):
    """Process feature extraction and save results to a CSV file."""
    print(f"Loading input file: {input_file}")

    try:
        df = pd.read_csv(input_file, sep="\t")
        print(f"Input file loaded: {df.shape}")

        df_descriptors = df["canonical_smiles"].apply(FeatureProcessor.compute_descriptors).apply(pd.Series)
        df_final = df[['canonical_smiles', 'source']]  # Keep only required columns
        df_descriptors = df["canonical_smiles"].apply(FeatureProcessor.compute_descriptors).apply(pd.Series)
        df_final = pd.concat([df_final, df_descriptors], axis=1)


        df_final.to_csv(output_file, index=False)
        if os.path.exists(output_file):
            print(f"Feature extraction complete. Saved to: {output_file}")
        else:
            print(f"Feature extraction failed. No output file created!")

    except Exception as e:
        print(f"Error processing features: {e}")

if __name__ == "__main__":
    import sys
    process_and_save_features(sys.argv[1], sys.argv[2])


