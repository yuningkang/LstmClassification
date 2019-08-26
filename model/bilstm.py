from model.CommonRNN import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self,args,vocab,embedding_weight):
        super(BiLSTM,self).__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight))
        embedding_dim = embedding_weight.shape[1]
        self._lstm = RNN(input_size=embedding_dim,
                         hidden_size=args.hidden_size,
                         layer_num = args.num_layers,
                         batch_first=True,
                         bidirectional=True,
                         rnn_type='lstm'
                         )
        self._bidirectional = 2
        self._num_direction = 2 if  self._bidirectional else 1
        self._emb_dropout = nn.Dropout(args.dropout_emb)
        self._linear_dropout = nn.Dropout(args.dropout_linear)

        self._linear = nn.Linear(in_features=args.hidden_size*self._num_direction,
                                 out_features=vocab.tag_size)


    def self_attention(self,encoder_output,hidden):
        '''

        :param encoder_output:[seq_len,batch_size,hidden_size*num_direction]
        :param hidden: [batch_size,hidden_size*num_direction]
        :return:
        '''
        encoder_output = encoder_output.transpose(0,1)
        hidden = hidden.unsqueeze(dim=1)
        simulation = torch.bmm(hidden,encoder_output)
        simulation = simulation.squeeze(dim=1)
        #simlation of shape [batch_size,seq_len]
        att_weight = F.softmax(simulation,dim=1)
        # att_weight of shape [batch_size,seq_len]
        output = torch.bmm(att_weight.unsqueeze(dim=1),encoder_output).squeeze(dim=1)
        return  output




    def forward(self,inputs,mask):
        '''

        :param inputs: [batch_size,seq_len]
        :param mask: [batch_size,seq_len]
        :return:
        '''
        word_emb = self.word_emb(inputs)

        if self.training:
            self._emb_dropout(word_emb)
        #[batch_size,seq_len,hidden_size*num_derection]
        outputs,_ = self._lstm(word_emb,mask)
        outputs = outputs.transpose(1,2)
        #[batch_size,hidden_size*num_derection,1]
        outputs = F.max_pool1d(outputs,kernel_size = outputs.shape[-1]).squeeze(dim=2)
        #[batch_size,tag_size]
        outputs =  self._linear_dropout(outputs)

        logit = self._linear(outputs)

        return logit