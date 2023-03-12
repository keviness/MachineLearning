# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertForSequenceClassification

UNCASED = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
Model_cache_dir = '/Users/kevin/Desktop/program files/MeachineLearning/妇科用药推荐/FunctionsBert/Result/BestModel/'
#bert_config = BertConfig.from_pretrained(UNCASED)
#hidden_size = 768
#class_num = 22

class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.bert = BertModel.from_pretrained(UNCASED)
        #self.bert = BertForSequenceClassification.from_pretrained(UNCASED)
        self.drop = nn.Dropout(dropout)
        self.liner = nn.Linear(hidden_size, class_num)
        
    '''
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls = self.drop(outputs[1])
        out = F.sigmoid(self.fc(cls))
        return out
    '''
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        #print('output:\n',output)
        cls = self.drop(output[1])
        #print('cls:\n',cls.shape)
        output = torch.relu(self.liner(cls))
        #print('output:\n',output)
        return output
