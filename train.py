from dataloader.Dataloader import *
from config.config import *
import torch
from vocab.vocab import *
from model.CNNmodel import *
from Classifier import *

from model.bilstm import *

if __name__ == '__main__':
    np.random.seed(666)
    torch.manual_seed(666)
   # torch.cuda.maual_seed(666)
    opts = data_path_config('config/data_path.json')
    train_data = Dataloader(opts['data']['train_file'])
    dev_data = Dataloader(opts['data']['dev_file'])
    test_data = Dataloader(opts['data']['test_file'])
    args = arg_config()
    print(torch.cuda.is_available())

    # print(torch.backends.cudnn.enable)
    print(torch.cuda.device_count())
    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda)

    #创建词表
    vocab = creat_vocab(opts['data']['train_file'])
    #保存词向量
    vocab.save(opts['vocab']['save_vocab'])
    vec = vocab.get_emdedding_weight(opts['data']['embedding_weight'])
    model = CNN(args,vocab,vec).to(args.device)
    model_lstm = BiLSTM(args,vocab,vec)
    #classifier = Classifier(model, args, vocab)
    classifier = Classifier(model_lstm, args, vocab)
    classifier.train(train_data,dev_data,test_data,args.device)
    # for onebatch in dataslice(train_data,args):
    #     words,tag = batch_numberize(onebatch,vocab,args)
    #
    #     output = classifier.forward(words)
    #     print(output)
    #遍历word2id
    #print(vocab._word2id.items())
    #遍历id2word
    #print(vocab.id2word.items())
    #标签数
    #print(vocab.tag_size)
    #词汇数
    #print(vocab.vocab_size)








