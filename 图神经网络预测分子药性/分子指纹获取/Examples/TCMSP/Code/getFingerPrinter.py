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
import numpy as np

ExperimentDataPath = '/Users/kevin/Desktop/program files/研究生论文/寒热药性-化合物/Data Handle/Data/502experimentData.xlsx'

compFeaturePath = '/Users/kevin/Desktop/program files/研究生论文/寒热药性-化合物/Data Handle/ComponentInfoHandle/Data/IngredientsDescriptionSmiles.xlsx'

outputPath = '/Users/kevin/Desktop/program files/研究生论文/寒热药性-化合物/Data Handle/ComponentInfoHandle/Result/'
SheetName = 'Components'

def getData(compFeaturePath):
    compFeatureDataFrame = pd.read_excel(compFeaturePath)
    compNameArray = compFeatureDataFrame['name'].values
    smilesComponentArray = compFeatureDataFrame['smiles'].values
    print('compNameArray:\n', compNameArray)
    print('smilesComponentArray:\n', smilesComponentArray)
    compFeatureSelectDataFrame = compFeatureDataFrame.loc[:,'smiles':]
    return compNameArray, smilesComponentArray, compFeatureSelectDataFrame

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
    #getIngredientsSmilesInfo(IngredientsSelectedDataFrame, compSmilesPath)
    getMACCSPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'MACCS')
    getMorganPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Morgan')
    getTopologicalPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Topological')
    getAvalonPrinterInfo(compNameArray, smilesComponentArray, compFeatureSelectDataFrame, 'Avalon')
