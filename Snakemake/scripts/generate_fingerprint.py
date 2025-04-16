import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

class FingerprintGenerator:
    @staticmethod
    def generate_rdkit_fingerprints(smiles_list):
        """Generate Morgan fingerprints for a list of SMILES."""
        fingerprints = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fingerprints.append(fp)
            else:
                fingerprints.append(None)
        return fingerprints

def generate_fingerprints(input_file, output_file):
    """Generate RDKit fingerprints and save to a CSV file."""
    print(f"Loading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"Input file loaded: {df.shape}")

        # Generate fingerprints
        fingerprints = FingerprintGenerator.generate_rdkit_fingerprints(df["canonical_smiles"].tolist())

        # Convert fingerprints to bit vectors for saving
        fingerprint_bits = [list(fp) if fp else None for fp in fingerprints]

        # Add fingerprints to DataFrame
        df = df[['canonical_smiles']]  # Keep only canonical_smiles
        df["fingerprints"] = fingerprint_bits


        # Save output
        df.to_csv(output_file, index=False)
        print(f"Fingerprint generation complete. Saved to: {output_file}")
    except Exception as e:
        print(f"Error generating fingerprints: {e}")

if __name__ == "__main__":
    import sys
    generate_fingerprints(sys.argv[1], sys.argv[2])
