import shap
import xgboost as xgb
from xgboost import XGBClassifier
shap.initjs()  
import matplotlib
# 我们先训练好一个XGBoost model
X,y = shap.datasets.boston()
print('X:\n', X)
print('y:\n', y.shape)



model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)

#result = model.predict(xgb.DMatrix(X.iloc[:12,:]))
#print('result:\n', result)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值

#单个prediction的解释
#SHAP提供极其强大的数据可视化功能，来展示模型或预测的解释结果。
# 可视化第一个prediction的解释   
# 如果不想用JS,传入matplotlib=True
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)

#多个预测的解释
#如果对多个样本进行解释，将上述形式旋转90度然后水平并排放置，我们可以看到整个数据集的explanations ：
#上图的"explanation"展示了每个特征都各自有其贡献，将模型的预测结果从基本值(base value)推动到最终的取值(model output)；将预测推高的特征用红色表示，将预测推低的特征用蓝色表示
XX = shap.plots.force(explainer.expected_value, shap_values, X)
shap.save_html('xx.html', XX)
#plt.show()
#基本值(base_value)是我们传入数据集上模型预测值的均值，可以通过自己计算来验证：
y_base = explainer.expected_value
print('y_base:\n',y_base)

pred = model.predict(xgb.DMatrix(X))
print('pred:\n',pred.mean())


#summary_plot
#summary plot 为每个样本绘制其每个特征的SHAP值，这可以更好地理解整体模式，并允许发现预测异常值。每一行代表一个特征，横坐标为SHAP值。一个点代表一个样本，颜色表示特征值(红色高，蓝色低)。比如，这张图表明LSTAT特征较高的取值会降低预测的房价

# summarize the effects of all the features
shap.summary_plot(shap_values, X)

#Feature Importance：
#之前提到传统的importance的计算方法效果不好，SHAP提供了另一种计算特征重要性的思路。
#取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图(multi-class则生成堆叠的条形图)
shap.summary_plot(shap_values, X, plot_type="bar")

#Interaction Values
#interaction value是将SHAP值推广到更高阶交互的一种方法。树模型实现了快速、精确的两两交互计算，这将为每个预测返回一个矩阵，其中主要影响在对角线上，交互影响在对角线外。这些数值往往揭示了有趣的隐藏关系(交互作用)

shap_interaction_values = explainer.shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)

#dependence_plot
#为了理解单个feature如何影响模型的输出，我们可以将该feature的SHAP值与数据集中所有样本的feature值进行比较。由于SHAP值表示一个feature对模型输出中的变动量的贡献，下面的图表示随着特征RM变化的预测房价(output)的变化。单一RM(特征)值垂直方向上的色散表示与其他特征的相互作用，为了帮助揭示这些交互作用，“dependence_plot函数”自动选择另一个用于着色的feature。在这个案例中，RAD特征着色强调了RM(每栋房屋的平均房间数)对RAD值较高地区的房价影响较小。

# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("RM", shap_values, X)