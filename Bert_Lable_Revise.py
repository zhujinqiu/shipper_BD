import time
from BERT import FocalLoss
from BERT import Bert
import os
import random
import numpy as np
import pandas as pd
# from transformers import WEIGHTS_NAME, CONFIG_NAME
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import seaborn as sns
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import BertPreTrainedModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, BertModel, AutoModel
from sklearn import preprocessing,metrics
# from pytorch_pretrained import BertModel, BertTokenizer
from torch.nn import functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM,get_linear_schedule_with_warmup
# from transformers.
from torch.autograd import Variable
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
import collections
history = collections.defaultdict(list) # 记录每一折的各种指标
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


class InputDataset(object):
    def __init__(self, encodings, label):
        self.encodings = encodings
        self.label = label

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(int(self.label[idx]))
        return item

    def __len__(self):
        return len(self.label)


class GetData(object):
    def __init__(self,data):
        self.data = data
        self.undefine_label = ["公司","公司企业","综合市场"]
        # self.data_define = self.data.loc[data.label.isin(self.undefine_label)]
        self.data_train = self.data.loc[~self.data.label.isin(self.undefine_label)] ##小类非公司企业的细粒度数据
        self.data_test =  self.data.loc[self.data.label.isin(self.undefine_label)] # 粗粒度数据
    def split(self,test_size):
        """数据切分"""
        train_ds,dev_ds,_,_ = train_test_split(self.data_train,self.data_train["label_text"],test_size=test_size,random_state=1234)
        return train_ds,dev_ds,self.data_test

class TextEncoding(object):
    def __init__(self,max_len,batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")  # 预训练模型选取www.huggingface.co,BERT模型里一致
        self.max_len = max_len
        self.batch_size = batch_size

    def endoding_loader(self):
        """获取数据并分割"""
        train_ds,dev_ds,test = GetData(data).split(0.2)
        train_encoding = self.tokenizer(text=train_ds["name"].tolist(),
                                   truncation=True,
                                   max_length=self.max_len,  # max_length一般取最大文本字符串长度
                                   padding=True)
        dev_encoding = self.tokenizer(text=dev_ds["name"].tolist(),
                                 truncation=True,
                                 max_length=self.max_len,
                                 padding=True)
        test_encoding = self.tokenizer(text=test["name"].tolist(),
                                  truncation=True,
                                  max_length=self.max_len,
                                  padding=True)

        train_encoding = pd.DataFrame({"input_ids": train_encoding["input_ids"],
                                       "attention_mask": train_encoding["attention_mask"],
                                       "label": train_ds.label.tolist()})
        dev_encoding = pd.DataFrame({"input_ids": dev_encoding["input_ids"],
                                     "attention_mask": dev_encoding["attention_mask"],
                                     "label": dev_ds.label.tolist()})
        test_encoding = pd.DataFrame(
            {"input_ids": test_encoding["input_ids"], "attention_mask": test_encoding["attention_mask"]})
        train_dataset = InputDataset(train_encoding.iloc[:, :-1],
                                     train_encoding['label'].values,
                                     )
        dev_dataset = InputDataset(dev_encoding.iloc[:, :-1],
                                   dev_encoding['label'].values,
                                   )
        test_dataset = InputDataset(test_encoding, [0] * len(test_encoding))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # 构建trainloader
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader,dev_loader,test_loader

class Train_Val(object):
    def __init__(self,epochs,class_nums):
        self.train_loader,self.dev_loader,self.test_loader = TextEncoding(max_len=32,batch_size=32)
        self.class_nums = class_nums
        self.epochs = epochs
        self.total_steps = len(self.train_loader) * epochs  # 5= epochs total_steps*0.1
        self.lr_scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0,
                                                       num_training_steps=self.total_steps)


        pass

    def validation(self,dev_loader):
        model.eval()
        label_acc = 0
        pred_all = []
        label_all = []

        for batch in dev_loader:
            with torch.no_grad():
                # torch.cuda.empty_cache()
                input_ids = batch['input_ids'].to(device)  # gpu
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)  # gpu
                pred = model(
                    input_ids,
                    attention_mask
                )
                label_acc += (pred.argmax(1) == label).float().sum().item()
                predicted = pred.argmax(1)  # 预测标签
                pred_all.extend(list(predicted.cpu().numpy()))
                label_all.extend(list(label.cpu().numpy()))
        f1 = metrics.f1_score(pred_all, label_all, average='weighted')  # ,average='macro'
        label_acc = label_acc / len(dev_loader.dataset)
        return f1, label_acc

    def train(self):  # ,rdrop_coef
        """
        train_loader
        dev_loader
        k_fold 用于控制交叉验证
        """
        best_f1 = 0.0
        best_accuracy = 0.0
        total_train_loss = 0
        output_dir = "./"
        iter_num = 0
        total_iter = len(self.train_loader)
        for epoch in range(1, self.epochs + 1):
            model.train()
            print("========================epoch" + str(epoch) + "========================")
            for batch in tqdm(self.train_loader):
                # 正向传播
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)

                probs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(probs, label)
                iter_num += 1

                loss.backward()
                optim.step()
                # lr更新
                self.lr_scheduler.step()
                # torch.cuda.empty_cache()
                if iter_num % 30 == 0:  # 每30步打印结果

                    print("=iter_num" + str(iter_num) + "===" + f"loss: {loss.item()}")

                    f1, label_acc = self.validation(self.dev_loader)

                    ######写入日志tensorboard
                    # writer.add_scalar(tag="loss/train", scalar_value=loss.item(), global_step=iter_num)
                    # writer.add_scalar(tag="F1/val", scalar_value=f1, global_step=iter_num)
                    # writer.add_scalar(tag="acc/val", scalar_value=label_acc, global_step=iter_num)

                    #####或者使用collection.dic(list)记录结果
                    # history['iter_num'].append(iter_num)
                    # history['loss'].append(loss.item())
                    # history['f1'].append(f1)
                    # history['accuracy'].append(label_acc)

                    ##########

                    if (best_f1 <= f1 or (best_f1 == f1 and best_accuracy < label_acc)):
                        print("best f1 %.5f, update %.5f ---> %.5f" % (f1, best_f1, f1))
                        print("best_accuracy  %.5f, update %.5f ---> %.5f" % (label_acc, best_accuracy, label_acc))
                        best_f1 = f1
                        best_accuracy = label_acc
                        model._save(save_path='./')

                        torch.save(model, "./" + str(k_fold) + "ifoldbest_f1.pkl")

        f1, label_acc = self.validation(self.dev_loader)
        if (best_f1 <= f1 or (best_f1 == f1 and best_accuracy < label_acc)):
            print("best f1 %.5f, update %.5f ---> %.5f" % (f1, best_f1, f1))
            print("best_accuracy  %.5f, update %.5f ---> %.5f" % (label_acc, best_accuracy, label_acc))
            best_f1 = f1
            best_accuracy = label_acc
            # 保存模型
            torch.save(model, "./" + str(k_fold) + "ifoldbest_f1.pkl")
            model._save(save_path='./')
        # return best_f1, best_accuracy

if __name__ == '__main__':
    k_fold = 0
    data = pd.read_csv("processed.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_nums = data["label"].value_counts().shape[0]
    model = Bert(class_nums)
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=2e-5)
    loss_fn = FocalLoss(class_nums, torch.ones(class_nums) * 0.5)  # 定义损失函数
    Train_Val().train()





