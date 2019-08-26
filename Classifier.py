from dataloader.Dataloader import *
import  torch.nn as nn
import torch.optim
import time
class Classifier:
    def __init__(self,model,args,vocab):
        self._model = model
        self._vocab = vocab
        self._args = args

    # def forward(self,input):
    #     logit = self._model.forward(input)
    #
    #     return  logit
    #打印模型参数
    def summary(self):
        print(self._model)
    #训练
    def train(self,train_data,dev_data,test_data,args_device):
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=self._args.lr,
                                     weight_decay=self._args.weight_decay)
        #optimizer = torch.optim.SGD(self._model.parameters(),lr=self._args.lr,weight_decay=self._args.weight_decay)
        #optimizer = torch.optim.Adadelta(self._model.parameters(),lr=self._args.lr,weight_decay=self._args.weight_decay)
        patience = 3
        train_loss = 0
        train_acc = 0
        train_loss_list,train_acc_list = [],[]
        dev_loss_list, dev_acc_list = [], []
        for ep in range(self._args.epoch):
            start_time = time.time()
            #让模型进入训练模式
            self._model.train()

            for onebatch in get_batch(train_data,self._args.batch_size):
                # 数据变量化
                words,tags,mask = batch_numberize(onebatch,self._vocab,args_device)

                #梯度初始化（置0）
                self._model.zero_grad()
                #前向传播（数据喂给模型）
                pred = self._model(words,mask)
                #反向传播，计算误差
                loss = self.compuate_loss(tags,pred)
                train_loss += loss.data.item()
                train_acc += self.compuate_acc(tags,pred)
                loss.backward()
                #用梯度去更新参数
                optimizer.step()
            end_time = time.time()
            during_time = float(end_time - start_time)
            dev_acc,dev_loss = self.validate(dev_data, args_device)
            #test_loss,test_acc =self.evluate(test_data,args_device)

            train_acc /= len(train_data)

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            dev_acc_list.append(dev_acc)
            dev_loss_list.append(dev_loss)
            #epoch经过epoch轮，如果开发集acc没有上升或者loss没有下降，则停止训练
            # if (ep+1)% patience == 0 and dev_acc_list[ep]< dev_acc_list[ep-patience+1]:
            #     break

            print("[Epoch %d] train loss %.6f train_acc %.6f  Time%.5f" %(ep,train_loss,train_acc,during_time))
            print("[Epoch %d] dev loss %.6f dev_acc %.6f" % (ep, dev_loss, dev_acc))

        test_acc,test_loss = self.evluate(test_data, args_device)
        print("test_loss%.6f         test_acc%.6f" %(test_loss,test_acc))

    #验证
    def validate(self,dev_data,args_device):

        dev_loss = 0
        dev_acc = 0


        # 让模型进入评估模式
        self._model.eval()
        #with torch.no_grad:
        for onebatch in get_batch(dev_data, self._args.batch_size):
            # 数据变量化
            words, tags,mask = batch_numberize(onebatch, self._vocab, args_device)
            pred = self._model(words,mask)
            #计算误差
            loss = self.compuate_loss(tags, pred)
            dev_loss += loss.data.item()
            dev_acc += self.compuate_acc(tags, pred)


        dev_acc /= len(dev_data)


        return dev_acc,dev_loss
    #评估模型
    def evluate(self,test_data,args_device):
        test_loss = 0
        test_acc = 0
        test_loss_list, test_acc_list = [], []

        # 让模型进入评估模式
        self._model.eval()
        for onebatch in get_batch(test_data, self._args.batch_size):
            # 数据变量化
            words, tags,mask = batch_numberize(onebatch, self._vocab, args_device)
            pred = self._model(words,mask)
            # 计算误差
            loss = self.compuate_loss(tags, pred)
            test_loss += loss.data.item()
            test_acc += self.compuate_acc(tags, pred)

        test_acc /= len(test_data)

        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        return test_acc, test_loss
    #计算准确率
    def compuate_acc(self,true_tags,logit):
        #返回正确的item的数目,eq是返回一个矩阵，sum之后返回总数
        return  torch.eq(torch.argmax(logit,dim=1),true_tags).sum().item()


    #计算损失
    def compuate_loss(self, true_tags, logit):
        # CrossEntropyLoss = LogSoftmax + NLLoss
        loss = nn.CrossEntropyLoss()
        loss = loss(logit, true_tags)
        return loss




