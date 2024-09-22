import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fp(s, fp_dim=2048, pack=False):
    """
     Convert SMILES string to fingerprint bit vector. This is a convenience function to use when you want to convert a smiles fingerprint to a fingerprint bit vector.
     
     @param s: SMILES string to convert.
     @param fp_dim: number of bits to use in the fingerprint
    """
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1

    # Pack the array into a packed bit array.
    if pack:
        arr = np.packbits(arr)

    return arr

def batch_smiles_to_fp(s_list, fp_dim):
    """
    Converts a batch of SMILES strings to fingerprints.

    @param s_list: List of SMILES strings.
    @param fp_dim: Dimension of fingerprints to return.
    @return: A NumPy array of fingerprints in the same order as the SMILES strings
    """
    fps = []
    # Convert each SMILES string to a fingerprint and add it to the list
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)

    # Assert that the shape of the array matches the expected dimensions
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim

    return fps