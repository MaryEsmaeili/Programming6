import os
import dask.dataframe as dd

# Ensure paths are absolute
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(BASE_DIR, "data", "featurized")
output_file = os.path.join(BASE_DIR, "data", "featurized", "merged_features.csv")

def merge_features():
    """Merge all feature files into a single CSV using Dask."""
    
    # Ensure directory exists
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist.")
        return

    files = os.listdir(input_dir)
    if not files:
        print(f"No feature files found in {input_dir}! Exiting merge process.")
        return

    df = dd.read_csv(f"{input_dir}/*.csv").compute()
    
    # Ensure unique canonical_smiles and drop duplicate columns
    df = df.drop_duplicates(subset="canonical_smiles")
    
    df.to_csv(output_file, index=False)
    print(f"Merged features saved to {output_file}")

if __name__ == "__main__":
    merge_features()
