> 🔗 原文链接： [https://blog.csdn.net/weixin_482495...](https://blog.csdn.net/weixin_48249563/article/details/113923459)

> ⏰ 剪存时间：2022-11-30 01:10:05 (UTC+8)

> ✂️ 本文档由 [飞书剪存 ](https://www.feishu.cn/hc/zh-CN/articles/606278856233?from=in_ccm_clip_doc)一键生成

# 利用sklearn进行one-hot编码（LabelBinarizer与MultiLabelBinarizer）

众所周知，当我们遇到nominal型特征时（ [统计学 ](https://so.csdn.net/so/search?q=%E7%BB%9F%E8%AE%A1%E5%AD%A6&spm=1001.2101.3001.7020)上称为定类变量），也就是用文字或字符串进行表示而无大小顺序关系的变量，有时候需要将此类定类变量转换为定量变量（数值），从而进行下一步的数据分析或挖掘。

在 [sklearn ](https://so.csdn.net/so/search?q=sklearn&spm=1001.2101.3001.7020)中，有两个非常方便的class—— **LabelBinarizer **和  **MultiLabelBinarizer ** 。

针对单个nominal型特征，可以利用 **LabelBinarizer **可以快速进行one-hot编码，实现定类变量定量化。若存在多个nominal型特征，则使用  **MultiLabelBinarizer ** 。

话不多说，看代码：

```Python
import numpy as np
# 先创建一个特征
nominal = np.array([["A"],
                   ["B"],
                   ["C"],
                   ["D"]])
# 导入LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
one_hot = LabelBinarizer()  # 创建one-hot编码器
one_hot.fit_transform(nominal) # 对特征进行one-hot编码
```

```Python
# 转换前nominal
array([['A'],
       ['B'],
       ['C'],
       ['D']], dtype='<U1')
# 转换后结果
array([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1]])
```

三行代码解决one-hot编码，很快有没有！

对于多个nominal型特征的情况，操作也是类似的：

```Python
import numpy as np
# 创建多nominal
multi_nominal = np.array([["A","Black"],
                         ["B","White"],
                         ["C","Green"],
                         ["D","Red"]])
# 导入MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
multi_one_hot = MultiLabelBinarizer()
multi_one_hot.fit_transform(multi_nominal)
```

```Python
# 转换前结果
array([['A', 'Black'],
       ['B', 'White'],
       ['C', 'Green'],
       ['D', 'Red']], dtype='<U5')
# 转换后结果
array([[1, 0, 1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0]])
```
