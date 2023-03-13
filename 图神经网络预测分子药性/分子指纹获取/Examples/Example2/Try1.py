import os
from rdkit import Geometry
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

'''
FEAT = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
featfact = ChemicalFeatures.BuildFeatureFactory(FEAT)
mol = Chem.MolFromSmiles('c1cccnc1')
AllChem.EmbedMolecule(mol)
feats = featfact.GetFeaturesForMol(mol)
for feat in feats:
    print(feat.GetFamily())
    pos = feat.GetPos()
    print(pos.x, pos.y, pos.z)
    
'''
mol = Chem.MolFromSmiles( 'OCc1ccccc1CN' )
AllChem.EmbedMolecule(mol) #gen 3d
factory = Gobbi_Pharm2D.factory
#calc 3d p4 fp
fp3D = Generate.Gen2DFingerprint(mol, factory, dMat=Chem.Get3DDistanceMatrix(mol))
print('fp3D:\n', len(list(fp3D)))
#factory = Chem.Pharm2D.SigFactory
mol = Chem.MolFromSmiles( 'OCc1ccccc1CN' )
fp2D = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
#fp2D = Generate.Gen2DFingerprint(mol,factory)
print('fp2D:\n', len(list(fp2D)))
print(list(fp2D.GetOnBits())[:5])
print(list(fp3D.GetOnBits())[:5])
print(factory.GetBitDescription(1))