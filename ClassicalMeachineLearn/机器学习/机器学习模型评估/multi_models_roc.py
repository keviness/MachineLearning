def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
        """
        将多个机器模型的roc图输出到一张图上
        Args:
            names: list, 多个模型的名称
            sampling_methods: list, 多个模型的实例化对象
            save: 选择是否将结果保存（默认为png格式）
            
        Returns:
            返回图片对象plt
        """
        plt.figure(figsize=(12, 10), dpi=dpin)
        for (name, method, colorname) in zip(names, sampling_methods, colors):
            y_test_preds = method.predict(X_test)
            y_test_predprob = method.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
            plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
            plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
            plt.axis('square')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate',fontsize=10)
            plt.ylabel('True Positive Rate',fontsize=10)
            plt.title('ROC Curve',fontsize=10)
            plt.legend(loc='lower right',fontsize=8)
        if save:
            plt.savefig('/Users/kevin/Desktop/program files/研究生论文/期刊论文预试验/BestEstimators/Code/multi_models_roc.png')
        return plt
