import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from trajGAIL.preprocess.tool import decide_next_state
from trajGAIL.preprocess.helper import processing_state_features
from trajGAIL.evaluation import metrics
from trajGAIL.transformer.encoder import Encoder
import copy

STATE_DIM = 125

def random_sample_inputs(length, states):
    output = random.sample(states, length)
    return torch.from_numpy(np.asarray(output)).type(torch.FloatTensor)


def next_states(dircs, current_states, volume, train_airport, traffic):
    ret = torch.zeros((dircs.size()[0], STATE_DIM))
    for i, (dirc, current_state) in enumerate(zip(dircs, current_states)):
        next_state = decide_next_state(dirc, current_state) # a list
        next_state.extend(processing_state_features(next_state, volume, train_airport, traffic))
        ret[i, :] = torch.FloatTensor(next_state)
    return ret

class GenCNN(nn.Module):
    def __init__(self, state_dim, num_classes, embedding_dim, filter_sizes, num_filters, NSTEPS, volume, train_airport, traffic):
        super(GenCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(state_dim*NSTEPS, n, (f, embedding_dim)) for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(sum(num_filters), num_classes)
        
        self.volume = volume
        self.train_airport = train_airport
        self.traffic = traffic
        self.init_parameters()
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
    def forward(self, x):
        # input size: batch_size x NSTEPS x state_dim 
        emb = x.unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size x num_filter x seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size x num_filter]
        pred = torch.cat(pools, 1) # batch_size x num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway)*F.relu(highway) + (1. - torch.sigmoid(highway))*pred
        pred = torch.sigmoid(self.fc(self.dropout(pred)))
        return pred
    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action
    def targeting_prob(self, x, labels):
        action_prob = self.forward(x)
        return action_prob.gather(1, labels)
    def sample(self, inp, length, NSTEPS):
        states = []
        actions = []

        inp = inp.view(1, NSTEPS*5, 5, 5)

        for _ in range(length):
            action = self.select_action(inp)[0][0]
            next_state = decide_next_state(action.item(),
                                        (inp[0][-5][0][0].item(), inp[0][-5][0][1].item(), inp[0][-5][0][2].item(), inp[0][-5][0][3].item()))

            if (next_state != None) and ((next_state[0] not in range(1, 53)) or (next_state[1] not in range(1, 91))):
                action = torch.tensor(9)

            if next_state != None and next_state[2] > 289:
                action = torch.tensor(9)
            
            states.append(inp[0, -5:, :, :].view(-1))
            actions.append(action)
            if action.item() == 9:
                break

            next_state.extend(processing_state_features(next_state, self.volume, self.train_airport, self.traffic))
            temp = torch.zeros(1, NSTEPS*5, 5, 5)
            temp[0, :(NSTEPS-1)*5, :, :] = inp[0, 5:, :, :]
            temp[0, -5:, :, :] = torch.FloatTensor(next_state).view(5, 5, 5)
            inp = temp
            if len(states) >= length:
                break
        return torch.stack(states), torch.from_numpy(np.asarray(actions))

class GenLSTM(nn.Module):
    '''LSTM Generator'''
    def __init__(self, emb_dim, hidden_dim, vocab_size, max_seq_len, volume, train_airport, traffic, gpu=False):
        super(GenLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True) # input: (batch_size, seq_length, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()
        
        self.volume = volume
        self.train_airport = train_airport
        self.traffic = traffic
        self.init_params()

    def init_hidden(self, batch_size):
        '''
        Args: 
            h: (1, batch_size, hidden_dim) lstm hidden state
            c: (1, batch_size, hidden_dim) lstm cell state
        '''
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.gpu:
            h, c = h.cuda(), c.cuda()
        return h, c
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
    def forward(self, x, h, c):
        '''
        Args:
          x: (batch_size, 1, embedding_dim)
          h: (1, batch_size, hidden_dim), lstm hidden state
          c: (1, batch_size, hidden_dim), lstm cell state
        '''
        #print(x.size())
        x = x.view((-1, 1, self.emb_dim))
        output, (h, c) = self.lstm(x, (h, c))
        pred = F.log_softmax(self.fc(output.view(-1, self.hidden_dim)), dim=1)
        return pred, h, c
    def sample(self, num_samples, sample_from, x=None):
        samples = torch.zeros(num_samples, self.max_seq_len, self.emb_dim).type(torch.FloatTensor)
        sample_actions = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h, c = self.init_hidden(num_samples)

        if x is None:
            given_len = 0
        else:
            given_len = x.size(1)

        for i in range(given_len):
            out, h, c = self.forward(x[:, i, :], h, c)
            samples[:, i, :] = x[:, i, :]

        if x is None:
            ns = random_sample_inputs(num_samples, sample_from)     # sample initial states as input
            samples[:, 0, :] = ns
        else:
            out = torch.multinomial(torch.exp(out), 1)
            sample_actions[:, i] = out.view(-1).data
            if out[0].data == 9 or out[0].data == 19:
                return samples, sample_actions
            ns = next_states(out.view(-1).data, samples[:, i, :].data, self.volume, self.train_airport, self.traffic)

        if self.gpu:
            samples = samples.cuda()
            ns = ns.cuda()
        if given_len != 0 and given_len < self.max_seq_len:
            samples[:, i+1, :] = ns

        for i in range(given_len, self.max_seq_len):
            ns = ns.view(-1)
            out, h, c = self.forward(ns, h, c)               # out: num_samples x vocab_size
            acts = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            sample_actions[:, i] = acts.view(-1).data
            if acts[0].data == 9 or out[0].data == 19:
                return samples, sample_actions
            ns = next_states(acts.view(-1).data, samples[:, i, :].data, self.volume, self.train_airport, self.traffic)
            if self.gpu:
                ns = next_state.cuda()
            if i+1 < self.max_seq_len:
                samples[:, i+1, :] = ns

            ns = next_state.view(-1)
        return samples, sample_actions
    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting action sequence.
        Inputs: inp, target
            - inp: batch_size x seq_len x state_dimension
            - target: batch_size x seq_len
        """
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()[:2]
        inp = inp.permute(1, 0, 2)           # seq_len x batch_size x state_dime
        target = target.permute(1, 0)     # seq_len x batch_size
        h, c = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h, c = self.forward(inp[i], h, c)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward, lbd=0):
        """
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()[:2]
        inp = inp.permute(1, 0, 2)          # seq_len x batch_size x state_dim
        target = target.permute(1, 0)       # seq_len x batch_size
        h, c = self.init_hidden(batch_size)

        loss = 0
        entropy = 0

        for i in range(seq_len):
            out, h, c = self.forward(inp[i], h, c)
            for j in range(batch_size):
                loss += out[j][target.data[i][j]]*reward[j][i]     # log(P(y_t|Y_1:Y_{t-1})) * D
                entropy -= torch.exp(out[j][target.data[i][j]])*out[j][target.data[i][j]]
        return (loss + lbd*entropy)/batch_size


class GenAtt(nn.Module):
    '''Attention Generator'''
    def __init__(self, emb_dim, vocab_size, max_seq_len, N, heads, dropout, volume, train_airport, traffic, gpu=False):
        super(GenAtt, self).__init__()
        self.emb_dim = emb_dim
        self.gpu = gpu
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(emb_dim, max_seq_len, N, heads, dropout)
        self.fc = nn.Linear(emb_dim*max_seq_len, vocab_size)
        self.softmax = nn.LogSoftmax()
        
        self.volume = volume
        self.train_airport = train_airport
        self.traffic = traffic
        self.init_params()
    def init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else: 
                param.data.uniform_(-0.05, 0.05)
            if self.gpu:
                param = param.cuda()
    def forward(self, x, mask=None):
        '''
            x: (batch_size, max_seq_len, emb_dim)
        '''
        if self.gpu:
            x = x.cuda()
        e_outputs = self.encoder(x, mask)
        output = self.fc(e_outputs.view(x.size(0), -1))
        output = self.softmax(output)
        return output
    def sample(self, num_samples, sample_from, x=None):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        Outputs: samples, hidden
            - samples: num_samples x max_seq_length x state-action_dimension(a sampled sequence in each row)
        """
        samples = torch.zeros(num_samples, self.max_seq_len, self.emb_dim).type(torch.FloatTensor)
        sample_actions = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)
        given_len = 0

        if x is not None:
            given_len = x.size(1)
            for i in range(given_len):
                samples[:, i, :] = x[:, i, :]
            out = self.forward(samples)               # out: num_samples x vocab_size
            acts = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)

            sample_actions[:, i] = acts.view(-1).data
            inp = next_states(acts.view(-1).data, samples[:, i, :].data, self.volume, self.train_airport, self.traffic)
            if self.gpu:
                inp = inp.cuda()
            if given_len < self.max_seq_len:
                samples[:, i+1, :] = inp
        else:
            inp = random_sample_inputs(num_samples, sample_from)     # sample initial states as input
            samples[:, 0, :] = inp

            if self.gpu:
                samples = samples.cuda()
                inp = inp.cuda()

        for i in range(given_len, self.max_seq_len):
            out = self.forward(samples)               # out: num_samples x vocab_size
            acts = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)

            sample_actions[:, i] = acts.view(-1).data
            next_state = next_states(acts.view(-1).data, samples[:, i, :].data, self.volume, self.train_airport, self.traffic)
            if self.gpu:
                next_state = next_state.cuda()
            if i+1 < self.max_seq_len:
                samples[:, i+1, :] = next_state
        return samples, sample_actions
    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting action sequence.
        Inputs: inp, target
            - inp: batch_size x seq_len x state_dimension
            - target: batch_size x seq_len
        """
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()[:2]
        samples = torch.zeros((batch_size, self.max_seq_len, self.emb_dim)).type(torch.FloatTensor)
        sample_actions = torch.zeros((batch_size, self.max_seq_len)).type(torch.LongTensor)

        loss = 0
        for i in range(seq_len):
            samples[:, i, :] = inp[:, i, :]
            out = self.forward(samples)
            loss += loss_fn(out, target.data[:, i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward, lbd=0):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: inp, target
            - inp: batch_size x seq_len x state_dim
            - target: batch_size x seq_len
            - reward: batch_size 
        """
        batch_size, seq_len = inp.size()[:2]
        samples = torch.zeros((batch_size, self.max_seq_len, self.emb_dim)).type(torch.FloatTensor)
        sample_actions = torch.zeros((batch_size, self.max_seq_len)).type(torch.LongTensor)

        loss = 0
        entropy = 0
        for i in range(seq_len):
            samples[:, i, :] = inp[:, i, :]
            out = self.forward(samples)
            for j in range(batch_size):
                loss += out[j][target.data[j][i]]*reward[j]
                #loss += out[j][target.data[j][i].item()]*reward[j][i]     # log(P(y_t|Y_1:Y_{t-1})) * D
                entropy -= torch.exp(out[j][target.data[j][i]])*out[j][target.data[j][i]]

        return (loss + lbd*entropy)/batch_size

