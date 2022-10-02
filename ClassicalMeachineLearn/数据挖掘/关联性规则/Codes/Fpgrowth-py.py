from fpgrowth_py import fpgrowth

itemSetList = [['eggs', 'bacon', 'soup'],
                ['eggs', 'bacon', 'apple'],
                ['soup', 'bacon', 'banana']]
freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
print('freqItemSet:\n', freqItemSet)
print('rules:\n', rules) 

itemSetList = [[1, 2, 1],
                [1, 3, 1],
                [2, 2, 3]]
freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
print('freqItemSet:\n', freqItemSet)
print('rules:\n', rules) 
