import sys
import pandas as pd

class DataLoader:
    def __init__(self, synthetic_path, natural_path, output_path):
        self.synthetic_path = synthetic_path
        self.natural_path = natural_path
        self.output_path = output_path

    def load_data(self):
        try:
            columns = ["canonical_smiles"]  # Only keep canonical_smiles
            synthetic_df = pd.read_csv(self.synthetic_path, sep="\t", names=columns, header=0, engine='python')
            natural_df = pd.read_csv(self.natural_path, sep="\t", names=columns, header=0, engine='python')

            synthetic_df["source"] = "Synthetic"
            natural_df["source"] = "Natural"

            combined_df = pd.concat([synthetic_df, natural_df], ignore_index=True)

            print(f"Synthetic dataset shape: {synthetic_df.shape}")
            print(f"Natural dataset shape: {natural_df.shape}")
            print(f"Combined dataset shape: {combined_df.shape}")
            print("Columns after loading:", combined_df.columns)

            return combined_df
        except Exception as e:
            print(f"Error loading or processing files: {e}")
            return None

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 scripts/load_data.py <synthetic_file> <natural_file> <output_file>")
        sys.exit(1)

    synthetic_path = sys.argv[1]
    natural_path = sys.argv[2]
    output_path = sys.argv[3]

    loader = DataLoader(synthetic_path, natural_path, output_path)
    combined_df = loader.load_data()

    if combined_df is not None:
        combined_df.to_csv(output_path, sep="\t", index=False)
        print(f"Combined dataset saved to {output_path}")

if __name__ == "__main__":
    main()
