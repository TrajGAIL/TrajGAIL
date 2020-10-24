from __future__ import unicode_literals, print_function, division
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle
import time
import itertools
from matplotlib import pyplot as plt
import scipy.optimize
import matplotlib.cm as cm
from torch.autograd import Variable
import sys
import numpy as np
import random
from io import open
import torch
import torch.optim as optim
from torch import nn as nn, autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from math import ceil
from trajGAIL.preprocess.tool import judging_action, decide_next_state, state_action_separation
from trajGAIL.preprocess.paddata import PaddedTensorDataset, pad_sequences_states, pad_sequences_actions, create_dataset
from trajGAIL.preprocess import helper
from trajGAIL.preprocess.helper import batchwise_sample, batchwise_oracle_nll, prepare_discriminator_data
from trajGAIL.models.generator import *
from trajGAIL.models.discriminator import *

CUDA = True

VOCAB_SIZE = 20
MAX_SEQ_LEN = 45
BATCH_SIZE = 32

ADV_TRAIN_EPOCHS = 1000

STATE_DIM = 125
GEN_HIDDEN_DIM = 32
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
d_dropout = 0.75

gPGEPOCH = 20
dEPOCH = 10

learning_rate = 1e-7

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
ByteTensor = torch.ByteTensor 
ones = torch.ones
zeros = torch.zeros

'''state info'''
traffic = pickle.load(open('./Data/latest_traffic.pkl', 'rb'))
volume = pickle.load(open('./Data/latest_volume_pickups.pkl', 'rb'))
train_airport = pickle.load(open('./Data/train_airport.pkl', 'rb'))
trajectories = pickle.load(open('./Data/all_trajs.pkl', 'rb'))


def train_generator_PG(gen, gen_opt, dis, num_batches, rollout, model_type):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    for batch in range(num_batches):
        s, a = gen.sample(BATCH_SIZE*2, expert_st)        # 64 works best
        if model_type != 1:
            rewards = rollout.get_reward(s, 16, dis)
        else:
            rewards = rollout.get_reward(S, 1, dis)

        gen_opt.zero_grad()
        pg_loss = - gen.batchPGLoss(s, a, rewards)
        pg_loss.mean().backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = batchwise_oracle_nll(gen, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, expert_st, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs, pad_states):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using generator)
    pos_val = pad_states[:100]
    neg_val = generator.sample(100, expert_st)[0]
    val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s, a = batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE, expert_st)
        dis_inp, dis_target = prepare_discriminator_data(pad_states, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))


'''Measure the average negative log loss of samples data.'''
def train_NLLoss(gen, samples):
    total_loss = 0
    for i, (train_states, train_actions, train_lengths) in enumerate(samples):
        if CUDA:
            train_states = train_states.cuda()
            train_actions = train_actions.cuda()
            train_lengths = train_lengths.cuda()

        loss = gen.batchNLLLoss(train_states, train_actions)

        total_loss += loss.data.item()

    # each loss in a batch is loss per sample
    total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN
    print('The training NLLoss is: {}'.format(total_loss))
    return total_loss


if __name__ == "__main__":
    agent_id = int(sys.argv[1])
    model_type = int(sys.argv[2]) # 1: GAIL 2: LSTM-GAIL 3: TrajGAIL
    NSTEP = int(sys.argv[3])

    '''Get the length of trajectories'''
    exp_traj_length = np.zeros(len(trajectories[agent_id]))
    for i in range(len(trajectories[agent_id])):
        exp_traj_length[i] = len(trajectories[agent_id][i])

    train_states, train_actions = state_action_separation(trajectories[agent_id])

    # data_loader: seq_tensor, seq_lengths
    train_loader = create_dataset(train_states, train_actions, bs=16) 

    seq_lengths = torch.LongTensor([len(s) for s in train_states])
    pad_states = pad_sequences_states(train_states, seq_lengths)
    pad_actions = pad_sequences_actions(train_actions, seq_lengths)

    global expert_st
    expert_st = []
    for traj in trajectories[agent_id]:
        expert_st.append(traj[0][:125])

    global POS_NEG_SAMPLES 
    POS_NEG_SAMPLES = len(train_states)
    # MAIN
    if model_type == 1:                 
        gen = GenCNN(STATE_DIM, VOCAB_SIZE, GEN_HIDDEN_DIM, d_filter_sizes, d_num_filters, NSTEP, volume, train_airport, traffic,)
        dis = DisCNN(STATE_DIM*NSTEP, 1, VOCAB_SIZE, d_filter_sizes, d_num_filters, gpu=CUDA, dropout=d_dropout)
    elif model_type == 2:
        gen = GenLSTM(STATE_DIM, 32, VOCAB_SIZE, MAX_SEQ_LEN, volume, train_airport, traffic, gpu=CUDA)
        dis = DisLSTM(STATE_DIM, 32, 1, MAX_SEQ_LEN, gpu=CUDA)
    else: 
        gen = GenAtt(STATE_DIM, VOCAB_SIZE, MAX_SEQ_LEN, 1, 1, 0.2, volume, train_airport, traffic, gpu=CUDA)
        dis = DisAtt(STATE_DIM, 1, 32, MAX_SEQ_LEN, 1, 1, 0.2, gpu=CUDA)

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()

    gen_optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
    dis_optimizer = optim.Adam(dis.parameters(), lr=learning_rate)

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = batchwise_oracle_nll(gen, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, expert_st, gpu=CUDA)

    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    path1 = './Results/{}/{}/g_{}_d_{}_lr_{}/'.format(model_type, agent_id, gPGEPOCH, dEPOCH, learning_rate)
    if not os.path.exists(path1):
        os.makedirs(path1)

    rollout = Rollout(gen, 0.8, expert_st, CUDA)
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, dis, gPGEPOCH, rollout, model_type)
        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, train_loader, gen, 2, dEPOCH, pad_states)
    
        torch.save(gen.state_dict(), path1+'gen_e_{}.trc'.format(epoch))
        torch.save(dis.state_dict(), path1+'dis_e_{}.trc'.format(epoch))

