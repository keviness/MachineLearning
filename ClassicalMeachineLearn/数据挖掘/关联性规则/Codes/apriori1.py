import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

shopping_list = [['大豆','白菜'],
            ['白菜','尿布','葡萄酒','甜瓜'],
            ['大豆','尿布','葡萄酒','橙汁'],
            ['白菜','大豆','尿布','葡萄酒'],
            ['白菜','大豆','尿布','橙汁']]
# 转换成数据框
shopping_df = pd.DataFrame(shopping_list)
#print('shopping_df:\n', shopping_df)

shopping_Array = np.array(shopping_list)
#print('shopping_Array:\n', shopping_Array)

# 购物编码器
te = TransactionEncoder()
df_tf = te.fit_transform(shopping_Array)
#print('df_tf:\n', df_tf)
df = pd.DataFrame(df_tf,columns=te.columns_)
print('df:\n', df)

# 求频繁项集
frequent_itemsets = apriori(df,min_support=0.05,use_colnames=True)  # 定义最小支持度为0.05
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)
print('frequent_itemsets:\n', frequent_itemsets)

# 求关联规则
association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.9)
association_rule.sort_values(by='leverage',ascending=False, inplace=True)
print('association_rule:\n', association_rule)