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
    def __init__(self, input, sent_cnt, word_cnt, dim=1024):
        self.input = input
        self.embed = Embedder()
        self.zeros = torch.zeros((len(self.input), sent_cnt, word_cnt, dim))

    def pad(self):
        paralen = [len(para) for para in self.input]
        seqlen = [[len(line.split()) for line in para] for para in self.input]
        self.embed = self.embed(input)
        for i in range(len(seqlen)):
            m = 0
            for k in seqlen[i]:
                self.zeros[i][m][:k] = self.embed[i][m][:k]
                m += 1
        return self.zeros

# INPUT FORMAT:
# input = [["i won the match", "i lost it", "wdw fd india wins"],
#          ["india will always win", "australia will always lose", "rishabh is"]]
# a = Pad(input, 10, 30)
# b = a.pad()
# print(b)

class WordEncoder(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1024, keep_prob=0.2, X): #input_dim= input dim at each time step = emb_size
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.upw = Variable(torch.rand((1, X.shape[0]*X.shape[1]*X.shape[2], 1)), requires_grad=True)   #incomplete
        self.GRU = nn.GRU(input_dim, hidden_dim/2 , hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, X):
        x1, x2 = (X.shape[0]*X.shape[1]*X.shape[2], X.shape[3])   #(4*10*50, 1024)
        X = X.unsqueeze(0)                                     #(1, 4*10*50, 1024)
        out, h = self.GRU(X)
        upkji = self.tanh(self.fc(out))                    #u(p)(kji) = (1, 2000, 1024)
        # --------------line 65------
        num = torch.exp(self.upw*upkji)
        denom = torch.sum(torch.exp(self.upw*upkji), dim=1)
        alpha = num / denom
        first_level_attention = torch.sum(alpha*out, dim=1)        #(1,1024)
        return first_level_attention

class SentEncoder(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1024, keep_prob=0.2, X):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.GRU = nn.GRU(input_dim, hidden_dim, hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.ups = Variable(torch.rand((X.shape[0]*X.shape[1], X.shape[2], 1)), requires_grad=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.second_level_attention = WordEncoder(X)  ###lookout
        self.tanh = nn.Tanh()

    def forward(self, X):
        x1, x2, x3 = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3])
        X = X.view(x1, x2, x3)
        out, h = self.GRU(X)
        upkj = self.tanh(self.fc(out))  # u(p)(kj) = (40,50,1024)
        num = torch.exp(self.upw * upkj)
        denom = torch.sum(torch.exp(self.upw * upkj), dim=1)
        alpha = num / denom
        first_level_attention = torch.sum(alpha*out, dim=1)                                              #(40,1024)
        first_level_attention = first_level_attention.unsqueeze(0)                                   #(1,40,1024)
        second_level_attention = self.second_level_attention(first_level_attention) #(1,1024)
        return second_level_attention

class ParaEncoder(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1024, keep_prob=0.2, X):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.GRU = nn.GRU(input_dim, hidden_dim/2, hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.upp = Variable(torch.rand((X.shape[1], X.shape[2], 1)), requires_grad=True)   #change
        self.third_level_output = WordEncoder(X)
        self.second_level_output = SentEncoder(X)

    def forward(self, X):
        x1, x2, x3, x4 = (X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        first_level_output = torch.zeros(x1,x2,x4)    #(4,10,1024)
        for i in range(x1):  #for p-paragraphs
            out, h = self.GRU(X[i])                 #out = (10,50,1024)
            upk = self.tanh(self.fc(out))
            num = torch.exp(self.upp*upk)
            denom = torch.sum(torch.exp(self.upw * upk), dim=1)
            alpha = num / denom
            first_level_attention = torch.sum(alpha * out, dim=1)  # (10,1024)
            first_level_output[i] = first_level_attention
        #first_level_output = (4,10,1024)
        second_level_output = self.second_level_output(first_level_output)  #(4,1024)
        second_level_output = second_level_output.view(1,x1,x4)  #(1,4,1024)
        third_level_output = self.third_level_output(second_level_output)  #(1,1024)
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