import logging
from rdkit import Chem
import pandas as pd


class DataCleaner:
    def __init__(self, log_file="cleaning.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def is_utf8(cell):
        """Check if a string can be encoded as UTF-8."""
        try:
            if pd.isna(cell):
                return False
            cell.encode("utf-8")
            return True
        except Exception:
            return False

    @staticmethod
    def is_valid_smiles(smiles):
        """Check if a SMILES string is valid."""
        try:
            if pd.isna(smiles) or not isinstance(smiles, str):
                return False
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False

    @staticmethod 
    def clean(df):
        """
        Cleans the dataset by:
        - Removing duplicate rows
        - Dropping missing or invalid SMILES
        - Validating SMILES strings
        - Ensuring UTF-8 encoding for text fields
        - Returning only necessary columns
        """
        original_count = len(df)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Remove duplicates
        df = df.drop_duplicates()

        # Drop rows with missing or invalid SMILES
        df = df[df["canonical_smiles"].apply(DataCleaner.is_valid_smiles)]

        # Ensure UTF-8 encoding for all text fields
        for col in ["canonical_smiles"]:
            df = df[df[col].apply(DataCleaner.is_utf8)]

        df = df.reset_index(drop=True)

        # Select relevant columns
        clean_df = df[["canonical_smiles", "source"]].copy()

        # Logging
        cleaned_count = len(clean_df)
        deleted_count = original_count - cleaned_count
        logging.info(f"Cleaning complete: {cleaned_count} valid rows retained, {deleted_count} rows removed.")

        return clean_df


def main(input_file, output_file):
    logging.info(f"Cleaning dataset: {input_file}")

    df = pd.read_csv(input_file, sep="\t")
    clean_df = DataCleaner().clean(df)

    clean_df.to_csv(output_file, sep="\t", index=False)
    logging.info(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
