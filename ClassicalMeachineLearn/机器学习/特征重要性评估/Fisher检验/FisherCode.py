import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact

path = '/Users/kevin/Desktop/program files/研究生论文/期刊论文预试验/Fisher检验/Data/502experimentData.xlsx'
sheetName = 'Components'
outputPath = '/Users/kevin/Desktop/program files/研究生论文/期刊论文预试验/Fisher检验/Result/'

def getData(path):
    dataFrame = pd.read_excel(path, sheet_name=sheetName)
    #dataFrame.set_index('Property', inplace=True)
    MolIDSet = list(set(';'.join(dataFrame['MolID'].values).split(';')))
    MolIDSetArray = np.array(MolIDSet)
    HerbsArray = list(set(dataFrame['Herbs'].values))
    
    ColdDataFrame = dataFrame[dataFrame['Property']==0]
    #print('ColdDataFrame:\n', ColdDataFrame)
    coldHerbs = ';'.join(ColdDataFrame['Herbs'].values).split(';')
    coldMolID = ';'.join(ColdDataFrame['MolID'].values).split(';')
    coldMoleculeName = ';'.join(ColdDataFrame['MoleculeName'].values).split(';')
    coldMolIDArray = np.array(coldMolID)
    #print('coldMolIDArray:\n',coldMolIDArray)
    
    HotDataFrame = dataFrame[dataFrame['Property']==1]
    #print('HotDataFrame:\n', HotDataFrame)
    HotHerbs = ';'.join(HotDataFrame['Herbs'].values).split(';')
    HotMolID = ';'.join(HotDataFrame['MolID'].values).split(';')
    HotMoleculeName = ';'.join(HotDataFrame['MoleculeName'].values).split(';')
    hotMolIDArray = np.array(HotMolID)
    #print('hotMolIDArray:\n',Counter(hotMolIDArray))
    
    return coldMolIDArray, hotMolIDArray, MolIDSetArray, HerbsArray, coldHerbs, HotHerbs

def calculateMols(coldMolIDArray, hotMolIDArray, MolIDSetArray, coldHerbs, HotHerbs):
    pValues = []
    for mol in MolIDSetArray:
        coldHerbsChem = coldMolIDArray.tolist().count(mol)
        notInColdHerbsChem = len(coldHerbs) - coldHerbsChem
        #notInColdHerbs = len(coldMolID)-coldHerbs
        hotHerbsChem = hotMolIDArray.tolist().count(mol)
        notInHotHerbs = len(HotHerbs)-hotHerbsChem
        
        kf_data = np.array([[coldHerbsChem,notInColdHerbsChem],[hotHerbsChem,notInHotHerbs]])
        stats_kf = fisher_exact(kf_data)[1]
        #print('stats_kf:\n', stats_kf)
        pValues.append(stats_kf)
        print('coldHerbs:',coldHerbsChem, 'hotHerbs:',hotHerbsChem, 'notInColdHerbsChem:',notInColdHerbsChem, 'notInHotHerbs',notInHotHerbs)
    pValuesArray = np.array(pValues)
    pValueDataFrame = pd.DataFrame({'MolID':MolIDSetArray,               'p-value':pValuesArray})
    pValueDataFrame = pValueDataFrame[pValueDataFrame['p-value']<0.05].sort_values(by='p-value',ascending=True)
    pValueDataFrame.to_excel(outputPath+'FisherResult.xlsx', index=False)
    print('write to excel file successfully!')
    
def getVector(ComponentsList, ComponentSetList):
    VectorArray = []
    for ComponentList in ComponentsList:
        ComponentSet = np.zeros(len(ComponentSetList))
        for Component in ComponentList:
            if Component in ComponentSetList:
                ComponentSet[ComponentSetList.index(Component)] = 1
        VectorArray.append(ComponentSet)
    VectorArray = np.array(VectorArray)
    #print("VectorArray:\n",VectorArray)
    return VectorArray

if __name__ == '__main__':
    coldMolIDArray, hotMolIDArray, MolIDSetArray, HerbsArray, coldHerbs, HotHerbs = getData(path)
    calculateMols(coldMolIDArray, hotMolIDArray, MolIDSetArray, coldHerbs, HotHerbs)