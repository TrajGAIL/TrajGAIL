import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Different discriminator architectures'''

class DisCNN(nn.Module):
    ''' 
        A CNN discriminator architechture: Convolution >> Max-pooling >> Softmax
    '''
    def __init__(self, state_dim, num_classes, embedding_dim, vocab_size, filter_sizes, num_filters, gpu=False, dropout=0.2):
        super(DisCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(state_dim, n, (f, embedding_dim)) for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def forward(self, x):
        # input size: batch_size x seq_len x state_dim 
        emb = x.unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size x num_filter x seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size x num_filter]
        pred = torch.cat(pools, 1) # batch_size x num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway)*F.relu(highway) + (1. - torch.sigmoid(highway))*pred
        pred = torch.sigmoid(self.fc(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len x state_dim
        Returns: out
            - out: batch_size ([0,1] score)
        """
        out = self.forward(inp)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len x state_dim
            - target: batch_size (binary 1/0)
        """
        loss_fn = nn.BCELoss()
        out = self.forward(inp)
        return loss_fn(out, target)

class DisLSTM(nn.Module):
    '''A LSTM discriminator'''

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(DisGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((2*2*1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((2*2*1, batch_size, self.hidden_dim)))
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
        if self.gpu:
            h, c = h.cuda(), c.cuda()
        return h, c

    def forward(self, x, h, c):
        # input dim                                                # batch_size x seq_len x embedding_dim
        x = x.view((-1, 1, self.embedding_dim))
        output, (h, c) = self.lstm(x, (h, c))
        pred = self.softmax(self.fc(output.view(-1, self.hidden_dim)))
        return pred, h, c

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len x state_dim
        Returns: out
            - out: batch_size ([0,1] score)
        """
        h, c = self.init_hidden(inp.size()[0])
        out, _, _ = self.forward(inp, h)[0]
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len x state_dim
            - target: batch_size (binary 1/0)
        """
        loss_fn = nn.BCELoss()
        h, c = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, c)
        return loss_fn(out, target)

from trajGAIL.transformer.encoder import Encoder

class DisAtt(nn.Module):
    '''Discriminator with Attention'''
    def __init__(self, state_dim, num_classes, emb_dim, max_seq_len, N, heads, dropout, gpu=False):
        super(DisAtt, self).__init__()
        self.emb_dim = emb_dim
        self.gpu = gpu
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(emb_dim, max_seq_len, N, heads, dropout)
        self.fc = nn.Linear(emb_dim*max_seq_len, num_classes)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                param.data.uniform_(-0.05, 0.05)
            if self.gpu:
                param = param.cuda()

    def forward(self, x, mask=None):
        '''x: (batch_size, max_seq_len, emb_dim)'''
        if self.gpu:
            x = x.cuda()
        e_outputs = self.encoder(x, mask)
        output = self.fc(e_outputs.view(x.size(0), -1))
        return torch.sigmoid(output)


    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len x state_dim
        Returns: out
            - out: batch_size ([0,1] score)
        """
        out = self.forward(inp)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len x state_dim
            - target: batch_size (binary 1/0)
        """
        loss_fn = nn.BCELoss()
        out = self.forward(inp)
        return loss_fn(out, target)

'''Rollout '''
class Rollout(object):
    '''Rollout policy'''
    def __init__(self, model, update_rate, expert_st, CUDA):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.expert_st = expert_st
        self.CUDA = CUDA

    def get_reward(self, states, num, discriminator):
        '''Args:
            x: (batch_size x seq_len x embedding_dim)
            num: roll-out number
            discriminator: discriminator net
        '''
        rewards = Variable(torch.zeros(states.size()[:2]))
        if self.CUDA:
            rewards = rewards.cuda()
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        for i in range(num):
            for l in range(1, seq_len + 1):
                data = states[:, 0:l]
                samples, _ = self.own_model.sample(batch_size, self.expert_st, x=data)
                if self.CUDA:
                    samples = samples.cuda()
                pred = discriminator(samples)
                rewards[:, l-1] += torch.log(pred.view(-1))
        rewards /= (1.0 * num) # batch_size x seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

