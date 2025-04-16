import numpy as np
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit import Chem

class DescriptorFingerprintProcessor:
    @staticmethod
    def compute_avg_tanimoto_in_batches(fingerprints, batch_size=1000):
        """
        Compute the average Tanimoto similarity for each fingerprint using batch processing.
        """
        num_fps = len(fingerprints)
        tanimoto_sums = np.zeros(num_fps)
        counts = np.zeros(num_fps)

        for batch_start in range(0, num_fps, batch_size):
            batch_end = min(batch_start + batch_size, num_fps)
            batch_fps = fingerprints[batch_start:batch_end]

            for i, fp_i in enumerate(batch_fps, start=batch_start):
                for j, fp_j in enumerate(fingerprints):
                    if fp_i and fp_j and j > i:  # Avoid duplicate comparisons
                        similarity = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                        tanimoto_sums[i] += similarity
                        tanimoto_sums[j] += similarity
                        counts[i] += 1
                        counts[j] += 1

        avg_tanimoto = np.divide(tanimoto_sums, counts, out=np.zeros_like(tanimoto_sums), where=counts > 0)
        return avg_tanimoto

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

    @staticmethod
    def compute_tanimoto_distances(fingerprints):
        """Compute Tanimoto similarity matrix from fingerprints."""
        num_fps = len(fingerprints)
        tanimoto_matrix = np.zeros((num_fps, num_fps))
        for i in range(num_fps):
            for j in range(i, num_fps):
                if fingerprints[i] and fingerprints[j]:
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    tanimoto_matrix[i, j] = similarity
                    tanimoto_matrix[j, i] = similarity
        return tanimoto_matrix

    @staticmethod
    def compute_avg_tanimoto(fingerprints):
        """Compute the average Tanimoto similarity for each fingerprint."""
        tanimoto_matrix = DescriptorFingerprintProcessor.compute_tanimoto_distances(fingerprints)
        avg_tanimoto = tanimoto_matrix.mean(axis=1)
        return avg_tanimoto
