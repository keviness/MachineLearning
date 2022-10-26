#---------------P-R绘制-------------------
def PRPlot(estimator, name, testData, testLabels):
    #testScores = estimator.decision_function(testData)
    #AP = average_precision_score(testLabels, testScores, average='macro', pos_label=1, sample_weight=None)
    disp = plot_precision_recall_curve(estimator, testData, testLabels,name=name)
    disp.ax_.set_title('2-class Precision-Recall curve')
    disp.ax_.plot()
    plt.show()
    '''
    testScores = estimator.predict_proba(testData)
    print('testScores:\n', testScores)
    print('testLabels:\n', testLabels)
    precision, recall, thresholds = precision_recall_curve(testLabels, testScores)
    print('precision:\n', precision)
    print('recall:\n', recall)
    #计算AP
    AP = average_precision_score(testLabels, testScores, average='macro', pos_label=1, sample_weight=None)
    #print('AP:', AP)
    plt.figure("P-R Curve")
    plt.title(name+' P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision, label='AP='+str(AP))
    plt.legend(loc="lower right")
    plt.show()
    '''

def kFoldROC(estimator, dataArray, property, k, estimatorName, save):
    cv = StratifiedKFold(n_splits=k, shuffle=True)
    #classifier = SVC(kernel='linear',probability=True,random_state=666)
    classifier = estimator
    tprs = []
    aucs = []
    #balanced_accList = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
    for i, (train, test) in enumerate(cv.split(dataArray, property)):
        classifier.fit(dataArray[train], property[train])
        #pre_label = classifier.predict(dataArray[test])
        #true_label = property[test]
        #balanced_acc = balanced_accuracy_score(pre_label, true_label)
        #balanced_accList.append(balanced_acc)
        viz = plot_roc_curve(classifier, dataArray[test], property[test],
                             name='ROC fold {}'.format(i+1),
                             alpha=0.3, lw=1.5, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #ax.plot(mean_fpr[0:10],balanced_accList,color='g',label='balanced accurancy',alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color='grey',alpha=.2,label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05],ylim=[-0.05, 1.05],title=estimatorName+" ROC")
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    
    if save:
        plt.savefig(PROutputPath+estimatorName+'.png')
    plt.show()