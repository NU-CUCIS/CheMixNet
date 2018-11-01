from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, AtomPairs, MolFromSmiles
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect as TopologicalTorsionFingerPrint
from rdkit.Chem.Draw import IPythonConsole, SimilarityMaps
from rdkit.Chem.AtomPairs.Pairs import ExplainPairScore as ExplainAtomPairScore
from rdkit.Chem.Draw.SimilarityMaps import GetAPFingerprint as AtomPairFingerprint, GetTTFingerprint as TopologicalFingerprint,GetMorganFingerprint as MorganFingerprint


from ml_util import saveData

def getCountInfo(m, fpType):
#     m = Chem.MolFromSmiles(formula)
    fp = None
    if fpType=='AtomPair' or fpType.lower()=='atom':
        fp = Pairs.GetAtomPairFingerprint(m)
        return fp.GetNonzeroElements()
    elif fpType.lower()=='morgan' or fpType.lower()=='circular':
        fp = AllChem.GetMorganFingerprint(m,2)
        return fp.GetNonzeroElements()
    elif fpType=='Topological' or fpType.lower()=='topo':
        fp = Torsions.GetTopologicalTorsionFingerprint(m)
        Dict = fp.GetNonzeroElements()
        convertedDict = {}
        for elem in Dict:
            convertedDict[int(elem)] = Dict[elem]
        return convertedDict

def getKeys(mol,fpType):
    return getCountInfo(mol,fpType).keys()

def generateUnFoldedFingerprint(mols,fpType):
#     mols = []
#     for SMILE in SMILES:
#         mols += [MolFromSmiles(SMILE)]

    fpKeys,unfoldedFP = [],[]
    for mol in mols:
        fpKeys += getKeys(mol,fpType)
    fpKeys = list(set(fpKeys))

    ## Iterating over each molecule
    for mol in mols:
        fpDict = getCountInfo(mol,fpType)
        keys = fpDict.keys()
        x = list(set(fpKeys)-set(keys))
        y = [0]*len(x)
        dictionary = dict(zip(x,y))
        dictionary.update(fpDict)
        List = []
        for attribute in fpKeys:
            List += [dictionary[attribute]]
        unfoldedFP += [List]
    # saveData(fpKeys,fpType+'Positions')
    # print('saving positions to:'+fpType+'Positions.pkl')
    # saveData(unfoldedFP,fpType+'FoldedCount')
    # print('saving count to:'+fpType+'FoldedCount.pkl')
    return unfoldedFP, fpKeys

def getAtomPair(mol,nBits=1024):
	return SimilarityMaps.GetAPFingerprint(mol, fpType='bv',nBits=nBits)

def getTopological(mol,nBits=1024):
	return SimilarityMaps.GetTTFingerprint(mol, fpType='bv',nBits=nBits)

def getCircular(mol,nBits=1024):
    if nBits==1024:
        return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    else:
	    return SimilarityMaps.GetMorganFingerprint(mol, fpType='bv')

def getMACCS(mol):
	return MACCSkeys.GenMACCSKeys(mol)
