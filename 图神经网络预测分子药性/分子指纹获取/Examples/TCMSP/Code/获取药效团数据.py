import re
import pandas as pd
import os
import openpyxl
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
import numpy as np

ExperimentDataPath = '/Users/kevin/Desktop/program files/研究生论文/寒热药性-化合物/Data Handle/Data/502experimentData.xlsx'

compFeaturePath = '/Users/kevin/Desktop/program files/MeachineLearning/图神经网络预测分子药性/分子指纹获取/Examples/TCMSP/Data/ConcateDescriptionAvalonPrinter.xlsx'

outputPath = '/Users/kevin/Desktop/program files/MeachineLearning/图神经网络预测分子药性/分子指纹获取/Examples/TCMSP/Result/'
SheetName = 'Components'

def getData(compFeaturePath):
    compFeatureDataFrame = pd.read_excel(compFeaturePath)
    compNameArray = compFeatureDataFrame['name'].values
    smilesComponentArray = compFeatureDataFrame['smiles'].values
    print('compNameArray:\n', compNameArray)
    print('smilesComponentArray:\n', smilesComponentArray)
    compFeatureSelectDataFrame = compFeatureDataFrame.loc[:,'smiles':]
    return compNameArray, smilesComponentArray, compFeatureSelectDataFrame

def getFp2DPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        FEAT = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        featFactory = ChemicalFeatures.BuildFeatureFactory(FEAT)
        #print('featFactory:\n', featFactory)
        #sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
        sigFactory = SigFactory(featFactory)
        sigFactory.SetBins([(0,500)])
        sigFactory.Init()
        fp2D = Generate.Gen2DFingerprint(mol,sigFactory)
        fp2DString = fp2D.ToBitString()
        resultArray.append([comName, fp2DString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    #concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    fingerPrinterDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')


def getGobbiFp3DPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        #mol = Chem.MolFromSmiles( 'OCc1ccccc1CN' )
        #AllChem.EmbedMolecule(mol) #gen 3d
        #factory = Gobbi_Pharm2D.factory
        fp3D = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        #print('fp3D:\n', len(list(fp3D)))
        #MACCSPrinter = MACCSkeys.GenMACCSKeys(mol)
        #MACCSPrinterString = MACCSPrinter.ToBitString()
        fp3DString = fp3D.ToBitString()
        resultArray.append([comName, fp3DString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    #concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    fingerPrinterDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')

def getMACCSPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        MACCSPrinter = MACCSkeys.GenMACCSKeys(mol)
        MACCSPrinterString = MACCSPrinter.ToBitString()
        resultArray.append([comName, MACCSPrinterString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    concateDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')
    
def getMorganPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        Printer = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        PrinterString = Printer.ToBitString()
        resultArray.append([comName, PrinterString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    concateDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')

def getTopologicalPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        Printer = Chem.Fingerprints.FingerprintMols.FingerprintMol(mol)
        Printer = pyAvalonTools.GetAvalonFP(mol)
        PrinterString = Printer.ToBitString()
        resultArray.append([comName, PrinterString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    concateDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')

def getAvalonPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, name):
    resultArray = []
    for comName, smiles in zip(compNameArray, smilesComponentArray):
        mol = Chem.MolFromSmiles(smiles)
        Printer = pyAvalonTools.GetAvalonFP(mol)
        PrinterString = Printer.ToBitString()
        resultArray.append([comName, PrinterString])
    resultArray = np.array(resultArray)
    fingerPrinterDataFrame = pd.DataFrame(data={'name':resultArray[:,0],
                                                'fingerPrinter':resultArray[:,1]})
    concateDataFrame = pd.concat([fingerPrinterDataFrame, compFeatureSelectDataFrame], axis=1)
    concateDataFrame.to_excel(outputPath+'ConcateDescription'+name+'Printer.xlsx', index=False)
    print('write to excel file successfully!')
    
if __name__ == '__main__':
    compNameArray, smilesComponentArray, compFeatureSelectDataFrame = getData(compFeaturePath)
    
    '''
    #getIngredientsSmilesInfo(IngredientsSelectedDataFrame, compSmilesPath)
    getMACCSPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'MACCS')
    getMorganPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Morgan')
    getTopologicalPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Topological')
    getAvalonPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Avalon')
    '''
    #getGobbiFp3DPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'GobbiFp3D')
    getFp2DPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'FP2D')