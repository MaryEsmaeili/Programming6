import os
import dask.dataframe as dd

# Dynamically determine the correct path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(BASE_DIR, "data", "fingerprint")
output_file = os.path.join(BASE_DIR, "data", "fingerprint", "merged_fingerprints.csv")

def merge_fingerprints():
    """Merge all fingerprint files into a single CSV using Dask."""

    # Ensure directory exists
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist.")
        return

    files = os.listdir(input_dir)
    if not files:
        print(f"No fingerprint files found in {input_dir}! Exiting merge process.")
        return

    df = dd.read_csv(f"{input_dir}/*.csv").compute()
    if 'compound_name' in df.columns:
        df = df.drop_duplicates(subset="canonical_smiles")  # Keep unique compounds
        df.drop(columns=['compound_name'], inplace=True)

    df.to_csv(output_file, index=False)

    print(f"Merged fingerprints saved to {output_file}")

if __name__ == "__main__":
    merge_fingerprints()
