from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os
import logging

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
            
            top_natural_substitutes = substitutes_df.loc[substitutes_df.groupby("Natural_Compound")["Tanimoto_Similarity"].idxmax()]
            
            # Generate molecule images for both Natural and Synthetic compounds
            for _, row in top_natural_substitutes.iterrows():
                natural_smiles = row["Natural_Compound"]
                synthetic_smiles = row["Synthetic_Compound"]
                
                natural_vis = MoleculeVisualizer(natural_smiles)
                synthetic_vis = MoleculeVisualizer(synthetic_smiles)
                
                natural_image_path = os.path.join(output_dir, f"Natural_{row.name}.png")
                synthetic_image_path = os.path.join(output_dir, f"Synthetic_{row.name}.png")
                
                natural_vis.draw(save_path=natural_image_path)
                synthetic_vis.draw(save_path=synthetic_image_path)
            
            print("Top Unique Natural Substitutes with visualizations generated.")
            return top_natural_substitutes
        except FileNotFoundError:
            logging.error(f"File {substitutes_file} not found. Ensure that the substitutes report has been generated.")
        except Exception as e:
            logging.error(f"Error processing the substitutes report: {e}")
