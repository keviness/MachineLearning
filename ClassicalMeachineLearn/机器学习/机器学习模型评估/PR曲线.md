# 【python】使用sklearn画PR曲线，计算AP值

> 🔗 原文链接： [https://blog.csdn.net/weixin_424861...](https://blog.csdn.net/weixin_42486139/article/details/114922196)

```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
 
'''
y_true: 
  类型：np.array； gt标签
y_scores：
  类型：np.array； 由大至小排序的阈值score,
'''
#画曲线
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall,precision)
plt.show()
 
#计算AP
AP = average_precision_score(y_true, y_scores, average='macro', pos_label=1, sample_weight=None)
print('AP:', AP)
```

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjA2ZTM4ZWVkYTI2NjRmYmQ0OGQ3N2QzNTY2NzdmZGFfUlZ0Mm1YUG1BalhVaDJKc2VzMEpYaXhkV3VQbDVHZVVfVG9rZW46Ym94Y25Ld0RZc2VlR3R4bFZ4aUQwdEJlbFNmXzE2NjY3MDU2Nzk6MTY2NjcwOTI3OV9WNA)
