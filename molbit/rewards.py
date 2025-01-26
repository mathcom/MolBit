import numpy as np
import networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRings
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import QED
from molbit.sascorer import calculateScore # https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
 

def calc_qed(smi):
    try:
        mol = MolFromSmiles(smi)
        qed = QED.qed(mol)
    except:
        qed = 0.
    return qed
    
    
def calc_penalizedlogp(smi):
    try:
        mol = MolFromSmiles(smi)
        clogp = MolLogP(mol) 
        sascore = calculateScore(mol)
        numring = calc_num_long_cycles(mol)
        penalizedlogp = clogp - sascore - numring
    except:
        penalizedlogp = 0.
    return penalizedlogp



'''
Popova, Mariya, et al. "MolecularRNN: Generating realistic molecular graphs with optimized properties."
arXiv preprint arXiv:1905.13372 (2019).
'''
class RewardQED(object):
    def __init__(self):
        super(RewardQED, self).__init__()

    def __call__(self, smi, debug=False):
        reward = 0.
        qed = 0.
        try:
            mol = MolFromSmiles(smi)
            qed = QED.qed(mol) # quantitative estimation of drug-likeness: ranges between 0 and 1, with 1 being the most drug-like.
            reward = 10. * qed # reward.range: 0 ~ 10 
        except:
            reward = 0.
            qed = 0.
        if debug:
            return reward, qed
        else:
            return reward
        

'''
Popova, Mariya, et al. "MolecularRNN: Generating realistic molecular graphs with optimized properties."
arXiv preprint arXiv:1905.13372 (2019).

Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Junction tree variational autoencoder for molecular
graph generation." International conference on machine learning. PMLR, 2018.

https://github.com/google-research/google-research/blob/master/mol_dqn/experimental/max_qed_with_sim.py
'''
def calc_num_long_cycles(mol):
    """Calculate the number of long cycles.
        Args:
        mol: Molecule. A molecule.
        Returns:
        negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    
    num_rings = 0
    if cycle_list:
        for cycle in cycle_list:
            if len(cycle) > 6:
                num_rings += 1
    return num_rings


class RewardPenalizedLogP(object):
    def __init__(self):
        super(RewardPenalizedLogP, self).__init__()

    def __call__(self, smi, debug=False):
        reward = 0.
        clogp = 0.
        sascore = 0.
        numring = 0.
        penalizedlogp = 0.
        try:
            mol = MolFromSmiles(smi)
            clogp = MolLogP(mol) 
            sascore = calculateScore(mol)
            numring = calc_num_long_cycles(mol)
            penalizedlogp = clogp - sascore - numring
            reward = 5. * penalizedlogp
        except:
            reward = 0.
            clogp = 0.
            sascore = 0.
            numring = 0.
            penalizedlogp = 0.
        if debug:
            return reward, clogp, sascore, numring, penalizedlogp
        else:
            return reward


'''
Ertl, Peter, and Ansgar Schuffenhauer. "Estimation of synthetic accessibility score of
drug-like molecules based on molecular complexity and fragment contributions."
Journal of cheminformatics 1.1 (2009): 1-11.
'''
class RewardSAScore(object):
    def __init__(self):
        super(RewardSAScore, self).__init__()

    def __call__(self, smi):
        reward = 0.
        try:
            mol = MolFromSmiles(smi)
            sascore = calculateScore(mol) # synthetic accessibility as a score between 1 (easy to make) and 10 (very difficult to make)
            reward = 10. - sascore # reward.range: 0 ~ 9
        except:
            pass
        return reward