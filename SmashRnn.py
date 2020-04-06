import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import torch.optim as optimizer

g=tf.Graph()
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
with g.as_default():
    text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    elmo = hub.Module("https://tfhub.dev/google/elmo/2",trainable=False)
    embed_text=elmo(text_input,signature='default',as_dict=True)['elmo']
    init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()
session=tf.compat.v1.Session(graph=g,config=config)
session.run(init_op)

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        elmo_embed=torch.tensor([session.run(embed_text,feed_dict={text_input:i}) for i in X])
        return elmo_embed


class Pad():  #pads words as well as sentences
    def __init__(self, bs=2, sent_cnt=2, word_cnt=50, dim=1024):
        sent_cnt += 1
        self.embed = Embedder()
        self.zeros = torch.zeros((bs, sent_cnt, word_cnt, dim))
        self.dummy = "<PAD> " * word_cnt

    def __call__(self, input):
        for li in input:
            li.append(self.dummy)
        paralen = [len(para) for para in input]
        seqlen = [[len(line.split()) for line in para] for para in input]
        self.embed = self.embed(input)
        for i in range(len(seqlen)):
            m = 0
            for k in seqlen[i]:
                self.zeros[i][m][:k] = self.embed[i][m][:k]
                m += 1

        alpha = self.zeros[:, :-1]
        return alpha

# INPUT FORMAT:
# input = [["i won the match", "i lost it", "wdw fd india wins"],
#          ["india will always win", "australia will always lose", "rishabh is"]]
# a = Pad(input, 10, 30)
# b = a.pad()
# print(b)

class WordEncoder(nn.Module):
    def __init__(self,input_dim=1024, hidden_dim=1024, output_dim=1): #input_dim= input dim at each time step = emb_size
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.upw = nn.Linear(hidden_dim,output_dim)  #1024x1
        self.fc = nn.Linear(hidden_dim, hidden_dim) 
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, X):  # padded elmo embed
        x1, x2 = (X.shape[0]*X.shape[1]*X.shape[2], X.shape[3])   #(4*10*50, 1024)
        X = X.view(1,x1,-1)                                 #(1, 4*10*50, 1024)
        out = X
        upkji = self.tanh(self.fc(out))            #(bs,2000,1024)                       
        alpha = self.softmax(self.upw(upkji))   #(bs,2000,1)
        first_level_attention = torch.sum(alpha*out, dim=1,keepdim=True)        #(bs,1,1024)
        return first_level_attention

class SentEncoder(nn.Module):
    def __init__(self,input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.GRU = nn.GRU(input_dim,int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.upw = nn.Linear(hidden_dim,output_dim)
        self.ups = nn.Linear(hidden_dim,output_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim) 
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, X):
        x1, x2, x3 = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3])   # (4*10,50,1024)
        X = X.view(x1, x2, x3)
        out = X
        upkj = self.tanh(self.fc(out))  # u(p)(kj) = (bs,40,50,1024)
        alpha=self.softmax(self.upw(upkj))
        first_level_attention = torch.sum(alpha*out, dim=-2).unsqueeze(0)                          #(bs,40,1024)      
        out,_=self.GRU(first_level_attention)
        upk=self.tanh(self.fc1(out))
        alpha=self.softmax(self.ups(upk))
        second_level_attention=torch.sum(alpha*out,dim=-2)       
        return second_level_attention

class ParaEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.GRU_Sent = nn.GRU(input_dim, int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.GRU_para = nn.GRU(input_dim, int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.FC1=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
        self.FC2=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
        self.FC3=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
    def forward(self, X): #(4,10,50,1024)
        out=X #(4,10,50,1024)
        alpha_word=self.FC1(out)
        first_level_attention=torch.sum(alpha_word*out,dim=-2)
        first_level_attention,_=self.GRU_Sent(first_level_attention)
        alpha_sent=self.FC2(first_level_attention)
        second_level_attention=torch.sum(alpha_sent*first_level_attention,dim=-2).unsqueeze(0)
        second_level_attention,_=self.GRU_para(second_level_attention)
        alpha_para=self.FC3(second_level_attention)
        third_level_output=torch.sum(alpha_para*second_level_attention,dim=-2)
        return third_level_output

class MashRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.X1 = WordEncoder()
        self.X2 = SentEncoder()
        self.X3 = ParaEncoder()

    def forward(self,X):
        X1 = self.X1(X)   #(1,1024)
        X2 = self.X2(X)   #(1,1024)
        X3 = self.X3(X)   #(1,1024)
        mashRNN = torch.cat([X1,X2,X3], dim=1)   #(1, 3*1024)
        return mashRNN

class Siamese(nn.Module):   #Siamese (Unsupervised, we don't have labels)

    def __init__(self):
        super().__init__()
        self.mashRNN = MashRNN()
        self.linear = nn.Sequential(nn.Linear(2*3072, 1024), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        out1 = self.mashRNN(x1)
        out2 = self.mashRNN(x2)
        cat =  torch.cat([out1,out2], dim=1)
        out = self.linear(cat)
        out = self.out(out)
        return out

def train(X1, X2):
    net = Siamese()
    opt = optimizer.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    opt.zero_grad()
    output = net(X1,X2)
    loss = loss_fn(output, target)
    loss.backward()
    opt.step()


if __name__ == '__main__':
    epochs = 200
    for epoch in range(epochs):
        train()
