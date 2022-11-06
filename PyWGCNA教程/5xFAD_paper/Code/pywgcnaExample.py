import PyWGCNA
geneExp = '/Users/kevin/Desktop/program files/MeachineLearning/PyWGCNA教程/5xFAD_paper/Data/expressionList_sorted.csv'
outputPath = '/Users/kevin/Desktop/program files/MeachineLearning/PyWGCNA教程/5xFAD_paper/Result'

#模型初始化，读取数据
pyWGCNA_5xFAD = PyWGCNA.WGCNA(name='5xFAD', species='mouse',
                              geneExpPath=geneExp, 
                              save=True, outputPath=outputPath)
result = pyWGCNA_5xFAD.geneExpr.to_df().head(5)
print('result:\n', result)

#数据预处理
'''
PyWGCNA允许您轻松预处理数据，包括删除缺失值过多或样品中表达量非常低的基因（默认情况下，我们建议删除表达不超过1 TPM的基因），以及删除缺失值过多或不匹配的样本。请记住，您可以通过更改和TPMcutoffcut'''
pyWGCNA_5xFAD.preprocess()

#基因网络的构建和模块的鉴定
'''
PyWGCNA将网络构建和模块检测的所有步骤压缩到一个函数中，该函数称为：findModules
选择软阈值功率：网络拓扑分析
共表达相似性和邻接性
拓扑重叠矩阵
使用 TOM 进行群集
合并表达式配置文件非常相似的模块
'''
#pyWGCNA_5xFAD.findModules()

#将模块与外部信息相关联并识别重要基因
'''
PyWGCNA在确定功能模块后收集了一些重要的分析，包括：analyseWGCNA()
量化模块-特征关系
基因与性状和模块的关系
在开始分析之前，请记住不要忘记添加有关样本或基因的任何信息。
为了显示模块关系热图，PyWGCNA需要用户使用函数从元数据的Matplotib颜色中指示颜色。setMetadataColor()
您还可以选择您希望在模块特征基因热图中显示的数据性状'''
pyWGCNA_5xFAD.updateMetadata(path='5xFAD_paper/metaData', 
                             sep='\t')
# add color for metadata
pyWGCNA_5xFAD.setMetadataColor('Sex', {'Female': 'green',
                                       'Male': 'yellow'})
pyWGCNA_5xFAD.setMetadataColor('Genotype', {'5xFADWT': 'darkviolet',
                                            '5xFADHEMI': 'deeppink'})
pyWGCNA_5xFAD.setMetadataColor('Age', {'4mon': 'thistle',
                                       '8mon': 'plum',
                                       '12mon': 'violet',
                                       '18mon': 'purple'})
pyWGCNA_5xFAD.setMetadataColor('Tissue', {'Hippocampus': 'red',
                                          'Cortex': 'blue'})

geneList = PyWGCNA.getGeneList(dataset='mmusculus_gene_ensembl',
                               attributes=['ensembl_gene_id', 
                                           'external_gene_name', 
                                           'gene_biotype'])

pyWGCNA_5xFAD.analyseWGCNA(geneList=geneList)


#保存和加载您的太平洋岛屿论坛
'''
您可以使用 或 函数 保存或加载 PyWGCNA 对象。saveWGCNA()readWGCNA()
'''
pyWGCNA_5xFAD.saveWGCNA()


'''
您还可以使用函数加载您的 PyWGCNA 对象。readWGCNA()
'''
import sys
sys.path.insert(0, '/Users/nargesrezaie/Documents/MortazaviLab/PyWGCNA')

import PyWGCNA
pyWGCNA_5xFAD = PyWGCNA.readWGCNA("5xFAD.p")