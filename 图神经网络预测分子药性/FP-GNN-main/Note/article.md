## 聚焦｜FP-GNN：基于分子指纹和图神经网络的分子性质预测模型

2022年9月17日，华南理工大学王领老师团队[1]在Briefings in Bioinformatics上发表文章。作者提出了FP-GNN，一种基于分子指纹（fingerprint，FP）和图神经网络（graph neural networks，GNN）的分子性质预测模型，结合了分子指纹表示和基于图神经网络的分子图表示。

在药物性质预测、药物-靶标关联预测和药物-疾病关联预测这三类任务、多个药物发现相关的数据集上大量实验表明，FP-GNN有效地预测了分子性质。作者收集和整理的大量数据集，以及FP-GNN的代码，已在其GitHub上开源：

https://github.com/idrugLab/FP-GNN

背景

准确预测分子特性，如物理化学和生物活性特性，以及ADMET（absorption, distribution, metabolism, excretion and toxicity，也就是药物分子的吸收、分布、代谢、排泄和毒性）特性，仍然是分子设计的基本挑战，尤其是对药物设计和发现而言。本文提出了一种新的深度学习架构，称为FP-GNN，结合并同时从分子图和指纹中学习分子特性信息。FP-GNN不仅能够刻画原子节点的局部特征，而且通过将节点信息传递到邻域节点，和在特定任务中使用注意力机制的远程节点，有效学习到分子的全局特征。通过结合分子指纹，模型的鲁棒性得以进一步增强。

方法

对于基于图的分子表示，分子的原子和键被视为节点和边。图神经网络通过聚合邻域节点信息，用于分子图表示的学习。如图1所示，FP-GNN采用图注意力网络（graph attention networks，GAT）[2]学习分子图表示。GAT将多头自注意力模型推广到图数据上，通过掩膜自注意力机制，自适应地为邻域节点中的不同节点分配不同的权重，在建模图结构上取得了更大的普适性和更优秀的结果。

本文综合使用了三种指纹，MACCS指纹，药效团ErG指纹和PubChem指纹，来描述分子特征。MACCS指纹是一种基于子结构密钥的指纹，包含大多数原子性质，化学键不同拓扑结构的性质和原子邻域，对药物发现有重要意义。PubChem指纹也是基于子结构密钥的指纹，可以更广泛地涵盖各种化学结构。药效团ErG指纹是使用扩展还原图（extended reduce graph，ErG）[3]方法的2D药效团指纹，并应用药效团类型节点描述对分子进行编码。三种指纹被拼接输入全连接层中进行训练。

在分别得到基于GNN的分子表示和基于指纹的分子表示之后，FP-GNN将这两种表示进行拼接，输入全连接层中，以输出预测结果。在本研究中，使用了Hyperopt Python包[4]对超参数进行贝叶斯优化，包括GNN的dropout率、多头注意力模型的头数、注意力层的隐藏层大小、指纹网络的隐藏层大小和dropout率等。

![图片](http://inews.gtimg.com/newsapp_bt/0/15394109141/641)

图1. ABCD-GGNN模型图

结果

本文使用三类基准数据集对FP-GNN模型的性能进行了广泛评估。一是使用由[5]整理的13个与药物发现相关的性质预测数据集测试FP-GNN的性能，包括3个物理化学数据集（ESOL，FreeSolv和Lipophilicity），6个生物活性和生物物理数据集（MUV，HIV、BACE、PDBbind-C、PDBbund-R和PDBbind-F），以及4个生理学和毒理学数据集（BBBP，Tox21，SIDER和ClinTox）。二是使用化合物-靶标数据集LIT-PCBA[6]测试FP-GNN的性能，该数据集包括15个靶标上7844个确认为活性的化合物，以及407381个确认为非活性的化合物。三是使用14个乳腺细胞系表型筛查数据集[7]来评估FP-GNN预测疾病相关化合物分子的能力。这些任务包括了回归任务和分类任务，回归任务通过均方根误差进行评估（RMSE），而分类任务由ROC曲线（ROC-AUC）或PR曲线（PRC-AUC）下的面积进行评估。

在药物性质预测任务中，FP-GNN与MoleculeNet[5], ChemProp[8], Attentive FP[9], HRGCN[10]和XGBoost进行了对比。FP-GNN在多个分类任务的ROC-AUC指标上取得最高值，在多个回归任务的RMSE上取得最低值。

表1：不同方法在药物性质预测上的对比

![图片](http://inews.gtimg.com/newsapp_bt/0/15394109145/641)

在预测药物在不同靶标上是否有活性的二分类预测任务上，FP-GNN与朴素贝叶斯（NB），支持向量机（SVM），随机森林（RF），XGBoost，以及深度神经网络（DNN），图卷积网络（GCN），图注意力网络（GAT）等深度学习模型进行了对比。FP-GNN的ROC-AUC值超越了大多数模型。

![图片](http://inews.gtimg.com/newsapp_bt/0/15394109377/641)

图2. 不同方法在药物靶标活性预测上的对比

在预测药物在乳腺细胞系上是否具有抗癌活性的任务中，FP-GNN与Attentive FP[9], GAT, GCN, MPNN（消息传递神经网络）和XGBoost进行了对比，在多个分类任务的ROC-AUC指标上取得最高值。

![图片](http://inews.gtimg.com/newsapp_bt/0/15394109381/641)

表2. 不同方法在药物疾病关联预测上的对比

总结

在本研究中，作者提出了一种称为FP-GNN的神经网络，将基于分子图的图注意力网络与基于分子指纹的全连接网络耦合起来，生成更全面的分子表征。在药物性质预测、药物在不同靶标上的活性预测、药物在不同细胞系中的抗癌活性预测这三种任务的多个公共数据集上的性能表明，FP-GNN的模型表现出色。

参考资料

[1] Cai et al., FP-GNN: a versatile deep learning architecture for enhanced molecular property prediction, Brief Bioinform, 2022

[2] Velickovic et al., Graph attention networks, in ICLR, 2018

[3] Stief et al., ErG: 2D pharmacophore descriptions for scaffold hopping. J Chem Inf Model, 2006

[4] Bergstra et al., Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures, in ICML, 2013

[5] Wu et al., MoleculeNet: a benchmark for molecular machine learning, Chem Sci, 2018

[6] Nguyen et al., LIT-PCBA: an unbiased data set for machine learning and virtual screening, J Chem Inf Model, 2020

[7] He et al., Machine learning enables accurate and rapid prediction of active molecules against breast cancer cells, Front Pharmacol, 2021

[8] Yang et al., Analyzing learned molecular representations for property prediction, J Chem Inf Model, 2019

[9] Xiong et al. Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism, J Med Chem, 2020

[10] Wu et al., Hyperbolic relational graph convolution networks plus: a simple but highly efficient QSAR modeling method, Brief Bioinform, 2021
