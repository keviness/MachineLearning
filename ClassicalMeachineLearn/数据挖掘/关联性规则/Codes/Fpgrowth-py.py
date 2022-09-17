from fpgrowth_py import fpgrowth

itemSetList = [['eggs', 'bacon', 'soup'],
                ['eggs', 'bacon', 'apple'],
                ['soup', 'bacon', 'banana']]
freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
print('freqItemSet:\n', freqItemSet)
print('rules:\n', rules) 
