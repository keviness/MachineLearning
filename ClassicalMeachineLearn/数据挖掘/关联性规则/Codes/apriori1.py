import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

shopping_list = [[1,0,1,1],
                [1,0,0,1],
                [1,0,1,1],
                [1,0,0,1],
                [1,0,1,0]]
# 转换成数据框
#shopping_df = pd.DataFrame(shopping_list)
#print('shopping_df:\n', shopping_df)

shopping_Array = np.array(shopping_list)
print('shopping_Array:\n', shopping_Array)

# 购物编码器
#te = TransactionEncoder()
#df_tf = te.fit_transform(shopping_Array)
#print('df_tf:\n', df_tf)
df = pd.DataFrame(shopping_Array,columns=['f1','f2','f3', 'f4'])
print('df:\n', df)

# 求频繁项集
frequent_itemsets = apriori(df,min_support=0.05,use_colnames=True)  # 定义最小支持度为0.05
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)
print('frequent_itemsets:\n', frequent_itemsets)

# 求关联规则
association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.5)
association_rule.sort_values(by='leverage',ascending=False, inplace=True)
#association_rule = association_rule[association_rule['consequents']=='(f4)']
print('association_rule:\n', association_rule)