#通用RNN，可以实现LSTM，GRU
import torch.nn as nn
import torch
class RNN(nn.Module):
    def __init__(self,input_size,#输入特征数量(对应词向量维度)
                 hidden_size, #隐层的特征数量
                 layer_num,#层数
                 batch_first=False,#如果为Ture那么输出[batch_size, seq_len, feature],如果为false，那么输入的格式为[seq_len,batch,feature]
                 bidirectional = False,#是否是双向LSTM
                 dropout = 0.0, #层与层之间使用
                 rnn_type = 'lstm'#默认是LSTM模型
                 ):
        super(RNN,self).__init__()
        self._num_direction = 2 if bidirectional else 1
        self._batch_first = batch_first
        self._hidden_size = hidden_size
        self._num_layer = layer_num
        self._bidirectional = bidirectional
        self._rnn_type = rnn_type.upper()
        self._rnn =['RNN','GRU','LSTM']
        assert self._rnn_type in self._rnn

        if self._rnn_type == "RNN":
            self._rnn_cell = nn.RNNCell #只是拿到了类，还没有实例化，加括号实例化
        elif self._rnn_type == "GRU":
            self._rnn_cell = nn.GRUCell
        elif self._rnn_type == "LSTM":
            self._rnn_cell = nn.LSTMCell

        self._fw_cells = nn.ModuleList()
        self._bw_cells = nn.ModuleList()
        for layer_i in range(layer_num):
            layer_input_size =input_size if layer_i == 0 else hidden_size  * self._num_direction
            self._fw_cells.append(self._rnn_cell(layer_input_size,hidden_size))
        #self._fw_cell = self._rnn_cell(input_size,hiiden_size)
            if self._bidirectional:
                #self._bw_cell = self._rnn_cell(input_size,hiiden_size)
                self._bw_cells.append(self._rnn_cell(layer_input_size, hidden_size))
       # self._word_emb = nn.Embedding(vocab)

    def _forward(self,cell,inputs,init_hidden,mask):
        '''
        :param inputs:[seq_len,batch_size,input_size]
        :param init_hidden:[batch_size,hidden_size],如果是LSTM，那么init_hidden是元组 h_0, c_0），h_0 of shape (batch, hidden_size)，c_0 of shape (batch, hidden_size):
        :param mask:[seq_len,batch_size,hidden_size]
        :return:[seq_len,batch_size,hidden_size]
        '''
        seq_len = inputs.shape[0]
        fw_hidden_next = init_hidden
        output = []
        for xi in range(seq_len):
            if self._rnn_type =="LSTM":
                #LSTMcell,前向传播的
                #每一个input[xi]都是[batch，input_size]的形式，而每个fw_hidden_next为batch_size,hidden_size]
                #h_next，c_next of shape (batch, hidden_size)
                h_next,c_next = cell(inputs[xi], fw_hidden_next)
                #如果是LSTM需要分别过滤
                #init_hidden(h0,c0)
                h_next = h_next * mask[xi]+init_hidden[0]*(1-mask[xi])
                c_next = c_next*mask[xi]+init_hidden[1]*(1-mask[xi])
                fw_hidden_next = (h_next,c_next)
                output.append(h_next)

            else:
                #rnncell or grucell
                fw_hidden_next =cell(inputs[xi],fw_hidden_next)
                # 如果不是LSTM，直接过滤即可
                fw_hidden_next = fw_hidden_next * mask[xi]

                output.append(fw_hidden_next)
        return  torch.stack(tuple(output),dim=0),fw_hidden_next

    def _backword(self,cell,inputs,init_hidden,mask):
        '''
        :param inputs:[seq_len,batch_size,input_size]
        :param init_hidden:[batch_size,hidden_size],如果是LSTM，那么init_hidden是元组 h_0, c_0），h_0 of shape (batch, hidden_size)，c_0 of shape (batch, hidden_size):
        :param mask:[seq_len,batch_size,hidden_size]
        :return:[seq_len,batch_size,hidden_size]
        '''
        seq_len = inputs.shape[0]
        bw_hidden_next = init_hidden
        output = []
        for xi in reversed(range(seq_len)):
            if self._rnn_type == "LSTM":
                # LSTMcell,前向传播的
                # 每一个input[xi]都是[batch，input_size]的形式，而每个fw_hidden_next为batch_size,hidden_size]
                # h_next，c_next of shape (batch, hidden_size)
                h_next, c_next = cell(inputs[xi], bw_hidden_next)
                # 如果是LSTM需要分别过滤
                # init_hidden(h0,c0)
                h_next = h_next * mask[xi] + init_hidden[0] * (1 - mask[xi])
                c_next = c_next * mask[xi] + init_hidden[1] * (1 - mask[xi])
                bw_hidden_next = (h_next, c_next)
                output.reverse()
                output.append(h_next)

            else:
                # rnncell or grucell
                bw_hidden_next = cell(inputs[xi], bw_hidden_next)
                # 如果不是LSTM，直接过滤即可
                bw_hidden_next = bw_hidden_next * mask[xi]
                output.reverse()
                output.append(bw_hidden_next)

        return torch.stack(tuple(output), dim=0),bw_hidden_next



    def forward(self, inputs,mask,init_hidden=None):
        '''
        :param inputs: [batch_size,seq_len,input_size]
        :param mask:
        :param init_hidden:
        :return:
        '''
        if self._batch_first:
            inputs = inputs.transpose(0,1)
            mask = mask.transpose(0,1)
        #input转置后为[seq_len,batch_size,input_size]
        batch_size = inputs.shape[1]
        #[seq_len,batch_size]->[seq_len,batch_size,1]->[seq_len,batch_size,hidden_size]
        #mask.unsqueeze(dim=2).expand((-1,-1,self._hidden_size))
        mask = mask.unsqueeze(dim=2).expand((-1,-1,self._hidden_size))
        if init_hidden  is None:
            init_hidden = self._init_hidden(batch_size,inputs.device)

        hn = []
        cn = []
        hx = init_hidden
        #fw_out:[seq_len,batch_size,hidden_size]
        #fw_hidden[batch_size,hidden_size],fw_hidden为最后一轮隐层的输出
        for i in range(self._num_layer):

            fw_out,fw_hidden = self._forward(self._fw_cells[i],inputs,hx,mask)
            bw_out, bw_hidden = None,None
            if self._bidirectional:
                bw_out, bw_hidden = self._backword(self._bw_cells[i],inputs, init_hidden, mask)
            if self._rnn_type == 'LSTM':
                 hn.append(torch.cat((fw_hidden[0],bw_hidden[0]),dim=1) if self._bidirectional else fw_hidden[0])
                 cn.append(torch.cat((fw_hidden[1],bw_hidden[1]),dim=1) if self._bidirectional else fw_hidden[1])

            #RNN/GRU
            else:
                 hn.append(torch.cat((fw_hidden,bw_hidden),dim=1) if self._bidirectional else fw_hidden)

            #[seq_len,batch_size,num_derection*hidden_size]
            inputs = torch.cat((fw_out,bw_out),dim=2) if self._bidirectional else fw_out

        output = inputs.transpose(0, 1) if self._batch_first else inputs
        #output:[batch_size,seq_len,num_derection*hidden_size]
        hn = torch.stack(tuple(hn),dim=0)
            #hn,cn:[num_layer,batch_size,hidden_size*num_derection]
        if self._rnn_type == "LSTM":
              cn  = torch.stack(tuple(cn),dim=0)
              hidden = (hn,cn)
        else:

              hidden = hn

        return  output,hidden

    def _init_hidden(self,batch_size,device = torch.device("cpu")):

        h0 = torch.zeros(batch_size,self._hidden_size,device = device)

        if self._rnn_type =="LSTM" :

            return h0 , h0

        else:
            return  h0

