import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,args,vocab,vec):
        super(CNN,self).__init__()
        extword_size,embedding_dim = vec.shape
        #print(embedding_dim)

        #卷积层
        #kernel_size卷积核大小（滤波器），stride步长，in_channels词向量维度，out_channels卷积核的个数,记一次卷积后产生多少个结果
        #窗口大小
        self.window_size=[3,4,5]
        self.conv =nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=embedding_dim,out_channels=args.hidden_size,kernel_size=w,stride=1,padding=1),
                                   nn.ReLU(),#激活函数
                                   nn.AdaptiveMaxPool1d(output_size=1)#自适应的，只规定输出即可，不用规定输入
                                   )for w in self.window_size])#创建三个卷积层，卷积核大小分别为3,4,5
        #drop_out，设为0.5的意义为每个节点都有百分之五十的概率被置成零
        self.dropout_emd = nn.Dropout(args.dropout_emb)
        self.dropout_linear = nn.Dropout(args.dropout_linear)
        #嵌入层
        #输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数,输出： (N, W, embedding_dim)
        self.extword_emb = nn.Embedding(vocab.extwords_size,embedding_dim)
        #self.word_emb = nn.Embedding(vocab.words_size,embedding_dim)

        #将随机的词向量替换成预训练的词向量
        self.extword_emb.weight.data.copy_(torch.from_numpy(vec))
#        print(self.word_emb)
        ##self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(vocab.get_emdedding_weight))
        # 全连接层1(线性层)
        self.fnn = torch.nn.Linear(args.hidden_size*len(self.window_size),vocab.tag_size)

    #前向传播
    def forward(self, inputs):#[batch_size,seq_len]
        embed = self.extword_emb(inputs)#[batch_size,seq_len,emdeding_dim]
        #因为一维卷积是在最后维度上扫的，不应该在词向量的维度上进行卷积，故进行转置
        if self.training:
            embed = self.dropout_emd(embed)

        embed = embed.transpose(1,2)
        ##[batch_size,out_channels,con_out]->过poolling层后变为（poolling层只针对最后一个维度进行池化）[batch_size,3*out_channels,1]->[batch_size,3*conv_out]，conv_out为out_channel来设置的
        #在dim=2的位置有维度唯一的位置squeeze将之抹掉
        conv_out = torch.cat(tuple([conv(embed)for conv in self.conv]),dim=1).squeeze(dim=2)
        #线性层dropout
        if self.training:
            conv_out = self.dropout_linear(conv_out)
        #[batch_size,tag_size]
        logit = self.fnn(conv_out)

        return logit