import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import logging
logging.set_verbosity_error()
import torch.nn.functional as nnf
from bert_multilabel_cls import BertMultiLabelCls
from sklearn.metrics import accuracy_score

# ---一，参数设置---
#SEED = 123
device = 'cuda' if torch.cuda.is_available() else 'cpu'
UNCASED = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
Model_cache_dir = '/Users/kevin/Desktop/program files/研究生论文/药性-BERT/中华本草/MultiLabelBert/BestModel/'
bert_config = BertConfig.from_pretrained(UNCASED)
BATCH_SIZE = 30
learning_rate = 2e-5
weight_decay = 1e-2
epsilon = 1e-8
maxLength = 60
epochs = 20
hidden_size = 768
#class_num = 22

# ----二，读取文件及文件预处理----
path = "/Users/kevin/Desktop/program files/MeachineLearning/妇科用药推荐/Data Precess/Result/FuNvVectorInfo.csv"
outputPath = '/Users/kevin/Desktop/program files/MeachineLearning/妇科用药推荐/FunctionsBert/Result/'

# ---load Data and prepare handle---
def getData(path):
    sourceDataFrame = pd.read_csv(path)
    #print("sourceDataFrame:\n", sourceDataFrame)
    herbsArray = sourceDataFrame['方剂名称'].values
    contentArray = sourceDataFrame["主治"].values
    #print("sentences:\n", contentArray)
    labelIds = sourceDataFrame.loc[:,'大豆黄卷':].values 
    #print("targets:\n", labelIds.shape)
    labelIds = torch.tensor(labelIds, dtype=torch.float)
    labels = sourceDataFrame.columns[3:]
    print('labels:\n', labels)
    return herbsArray, labelIds, contentArray, labels

def writeToExcelFile(trainLossArray,trainAccuracyArray):
    dataFrame = pd.DataFrame({'trainLossArray':trainLossArray,
                              'trainAccuracyArray':trainAccuracyArray})
    dataFrame.to_excel(outputPath+'FourTrainLossAcc.xlsx')
    print('write to excel successfully!')

def convertTextToToken(contentArray, maxLength):
    tokenizer = BertTokenizer.from_pretrained(UNCASED)
    inputIds = []
    attentionMask = []
    token_type_ids = []
    for elementText in contentArray:
        token = tokenizer(elementText, add_special_tokens=True, padding='max_length', truncation=True, max_length=maxLength)
        inputIds.append(token['input_ids'])
        attentionMask.append(token['attention_mask'])
        token_type_ids.append(token['token_type_ids'])
    inputIds = torch.tensor(inputIds, dtype=torch.long)
    attentionMask = torch.tensor(attentionMask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    #print('inputIds:\n', inputIds)
    #print('attentionMask:\n', attentionMask)
    return inputIds, attentionMask, token_type_ids

# -----------划分数据集-----------
def trainTestSplit(labelIds, inputIds, attentionMask, token_type_ids):
    train_input_ids, test_input_ids, train_labels, test_labels, train_masks, test_masks, train_token_type_ids, test_token_type_ids = train_test_split(inputIds,labelIds,attentionMask, token_type_ids, random_state=666, test_size=0.2)
    
    #print("train_labels:\n", test_labels)
    train_data = TensorDataset(train_input_ids, train_masks, train_labels, train_token_type_ids)
    #train_dataloader = DataLoader(train_data,  batch_size=BATCH_SIZE)
    test_data = TensorDataset(test_input_ids, test_masks, test_labels, test_token_type_ids)
    #print("train_data_test_labels:\n", train_data.train_labels)
    return train_data, test_data

# -----------创建优化器-----------
##参数eps是为了 提高数值稳定性 而添加到分母的一个项(默认: 1e-8)
def getOptimizer(model, weight_decay, epsilon):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay' : weight_decay
        },
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay' : 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)
    return optimizer

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    #print('y_pred_tensor:\n', y_pred_tensor)
    y_true_tensor = y_true_tensor.cpu().int().numpy()
    #print('y_true_tensor:\n', y_true_tensor)
    return accuracy_score(y_true_tensor, y_pred_tensor)

# -----------训练模型-----------
def train(train_data, test_data, num_labels):
    # 参数设置
    batch_size = BATCH_SIZE

    # 获取到dataset
    train_dataset = train_data
    test_dataset = test_data
    #test_data = load_data('cnews/cnews.test.txt')

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #print('train_dataloader:\n', len(train_dataloader))
    #print('valid_dataloader:\n', len(valid_dataloader))
    
    # 初始化模型
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=num_labels)
    #model = BertForSequenceClassification.from_pretrained(UNCASED, num_labels=num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    #optimizer = getOptimizer(model, weight_decay, epsilon)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    
    best_acc = 0
    trainLossArray = []
    trainAccuracyArray = []
    testLossArray = []
    testAccuracyArray = []
    for epoch in range(1, epochs+1):
        losses = 0      # 损失
        accuracy = 0    # 准确率
        #训练
        model.train()
        train_bar = tqdm(train_dataloader)
        for train_input_ids, train_masks, train_labels, train_token_type_ids in train_bar:
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            output = model(
                input_ids=train_input_ids.to(device), 
                attention_mask=train_masks.to(device), 
                token_type_ids=train_token_type_ids.to(device)
                #labels=train_labels.to(device)
            )
            #outputs = model(ids, mask, token_type_ids)
            #print('outPutTrain:\n', output)
            #loss, predict = output[0], output[1]  #loss: 损失, logits: predict
            loss = loss_fn(output, train_labels)
            #print("loss:\n", loss)
            
            trainLossArray.append(loss.item())
            losses += loss.item()
            #print("losses:\n", losses)
            #print("predict:\n", predict)
            #pred_labels = torch.argmax(output, dim=1)   # 预测出的label
            acc =  get_acc_score(train_labels, output)
            #print('acc:\n', acc)
            accuracy += acc
            trainAccuracyArray.append(acc)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # 更新模型参数
            # 大于1的梯度将其设为1.0, 以防梯度爆炸
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        train_average_loss = losses / len(train_dataloader)
        train_average_acc = accuracy / len(train_dataloader)
        print('\tTrain ACC:',train_average_acc,'\tTrain Loss:',train_average_loss)

        # 验证
        model.eval()
        losses = 0      # 损失
        accuracy = 0    # 准确率
        valid_bar = tqdm(valid_dataloader)
        for test_input_ids, test_masks, test_labels, test_token_type_ids in valid_bar:
            valid_bar.set_description('Epoch %i test' % epoch)
            output = model(
                input_ids=test_input_ids.to(device),
                token_type_ids=test_token_type_ids.to(device),  
                attention_mask=test_masks.to(device) 
            )
            #print("outputTest:\n", output)        
            #predict = output[0]  # loss: 损失, logits: predict
            #print("predict:\n", predict)
            loss = loss_fn(output, test_labels)
            
            #print("loss:\n", loss)
            testLossArray.append(loss.item())
            acc =  get_acc_score(test_labels, output)
            accuracy += acc
            
            valid_bar.set_postfix(loss=loss.item(), acc=acc)
            testAccuracyArray.append(acc)

        test_average_loss = losses / len(valid_dataloader)
        test_average_acc = accuracy / len(valid_dataloader)
        print('\tTest ACC:',test_average_loss,'\tTest Loss:', test_average_acc)

    if train_average_acc > best_acc:
        best_acc = train_average_acc
        torch.save(model.state_dict(), Model_cache_dir+'FKHerbs_best_model.pkl')

    writeToExcelFile(trainLossArray,trainAccuracyArray)
 
# ------------预测------------
def predict(labels, inputTextArray):
    #bert_config = BertConfig.from_pretrained('bert-base-chinese')
    #初始化模型
    num_labels = len(labels)
    
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=num_labels)
    #model = BertForSequenceClassification.from_pretrained(UNCASED,num_labels=num_labels) 

    model.load_state_dict(torch.load(Model_cache_dir+'FKHerbs_best_model.pkl',map_location=torch.device('cpu')))
    model.to(device)
    
    #初始化分词器
    tokenizer = BertTokenizer.from_pretrained(UNCASED)

    resultArray = []
    for inputText in inputTextArray:
        text = inputText.replace('\n', '')
        token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=maxLength)
        input_ids = token['input_ids']
        attention_mask = token['attention_mask']
        token_type_ids = token['token_type_ids']

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device)
        )
        
        print('output:\n', output)
        predicted = output[0]
        prob = nnf.relu(output[0]).reshape(-1).detach().numpy()
        probList = list(prob)
        #print('predicted:\n', predicted.detach().numpy())
        #print('probabilty:\n',prob)
        pred_label_id = torch.argmax(predicted, dim=1).detach().numpy()[0]
        #print("pred_label_id:\n", pred_label_id)
        predResult = labels[pred_label_id]
        #print('predResultLabel:', predResult)
        probList.insert(0, predResult)
        probList.insert(1, pred_label_id)
        #print('reluResult:\n', probList)
        resultArray.append(probList)
    resultArray = np.array(resultArray)
    return resultArray

def writeReluToExcel(HerbsArray, reluArray, columnsList):
    dataFrame = pd.DataFrame(data=reluArray, columns=columnsList)
    dataFrame.insert(0,'Herb', HerbsArray)
    dataFrame.to_excel(outputPath+'ZHBCHerbNew22PropertyReluResult.xlsx', index=False)
    
    print('Write to Excel file successfully!')

if __name__ == '__main__':
    #labels = ['寒','热','温','凉','酸','苦','甘','辛','咸','心','肝','胆','脾','胃','肺','肾','大肠','小肠','心包','膀胱','三焦','毒']
    
    #num_labels = 5
    #trian
    herbsArray, labelIds, contentArray, labels = getData(path)
    num_labels = len(labels)
    inputIds, attentionMask, token_type_ids = convertTextToToken(contentArray, maxLength)
    train_data, test_data = trainTestSplit(labelIds, inputIds, attentionMask, token_type_ids)
    
    train(train_data, test_data, num_labels)

    #predict
    #reluArray = predict(labels, contentArray)
    #writeReluToExcel(herbsArray, reluArray)
