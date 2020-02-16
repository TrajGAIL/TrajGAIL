import torch
from torch.autograd import Variable
from math import ceil
from trajGAIL.preprocess.tool import judging_action

def processing_state_features(input_state, volume, train_airport, traffic):
    '''
    input_state = (x, y, t, day)
    output = [5*5, 5*5, 5*5, 5*5, 5*5]
    '''
    x = int(input_state[0])
    y = int(input_state[1])
    t = int(input_state[2])
    day = int(input_state[3])

    if x == y == t == day == 0:
        return [0. for x in range(121)]

    x_range = list(range(x-2, x+3))
    y_range = list(range(y-2, y+3))
    
    # state features
    n_p = []
    n_v = []
    t_s = []
    t_w = []
    for i in x_range:
        for j in y_range:
            if (i, j, t, day) in volume:
                n_p.append(volume[(i, j, t, day)][0])
                n_v.append(volume[(i, j, t, day)][1])
            else:
                n_p.append(0.)
                n_v.append(0.)
                
            if (i, j, t, day) in traffic:
                t_s.append(traffic[(i, j, t, day)][0])
                t_w.append(traffic[(i, j, t, day)][1])
            else:
                t_s.append(0.)
                t_w.append(0.)
                
    ta = []
    for place in train_airport:
        ta.append(abs(x - train_airport[place][0][0]) + abs(y - train_airport[place][0][1]))
    
    whole_step = []
    whole_step.extend(ta)
    whole_step.extend(n_p)
    whole_step.extend(n_v)
    whole_step.extend(t_s)
    whole_step.extend(t_w)
    return whole_step

def process_trajectories(trajectories_dict, target_driver_no, drivers_reverse, volume, train_airport, traffic):
    outputs = []
    for traj in trajectories_dict[drivers_reverse[target_driver_no]]:
        temp_traj = []
        for i in range(len(traj)-1):
            step = traj[i].copy()
            new_step = step[1:].copy()

            # extract state features
            state_feature = processing_state_features(new_step, volume, train_airport, traffic)

            # find action
            act = judging_action(step[1], step[2], step[3], traj[i+1][1], traj[i+1][2], traj[i+1][3])
            ind = act

            new_step.extend(state_feature) # x, y, t, features...
            new_step.append(ind) # x, y, t, features, act

            temp_traj.append(new_step)
            step = 0
        last_step = traj[-1][1:].copy()
        state_feature = processing_state_features(last_step, volume, train_airport, traffic)
        last_step.extend(state_feature) # x, y, t, features 
        last_step.append(9) # x, y, t, features, act
        temp_traj.append(last_step)
        outputs.append(temp_traj)
    return outputs


def process_trajectory(trajectories, target_driver_no, drivers_reverse, volume, train_airport, traffic):
    '''No action is added'''
    outputs = []
    for traj in trajectories:
        temp_traj = []
        for i in range(len(traj)-1):
            step = traj[i].copy()
            new_step = step[1:].copy()

            # extract state features
            state_feature = processing_state_features(new_step, volume, train_airport, traffic)

            # find action
            act = judging_action(step[1], step[2], step[3], traj[i+1][1], traj[i+1][2], traj[i+1][3])
            ind = act

            new_step.extend(state_feature) # x, y, t, features...
            new_step.append(ind) # x, y, t, features, act

            temp_traj.append(new_step)
            step = 0
        last_step = traj[-1][1:].copy()
        state_feature = processing_state_features(last_step, volume, train_airport, traffic)
        last_step.extend(state_feature) # x, y, t, features 
        last_step.append(9)
        temp_traj.append(last_step)
        outputs.append(temp_traj)
    return outputs

def batchwise_sample(gen, num_samples, batch_size, sfrom):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    sample_states = []
    sample_actions = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        states, actions = gen.sample(batch_size, sfrom)
        sample_states.append(states)
        sample_actions.append(actions)

    return torch.cat(sample_states, 0)[:num_samples], torch.cat(sample_actions, 0)[:num_samples]


def batchwise_oracle_nll(gen, num_samples, batch_size, max_seq_len, sfrom, gpu=False):
    s, a = batchwise_sample(gen, num_samples, batch_size, sfrom)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        if gpu:
            inp, target = s[i:i+batch_size].cuda(), a[i:i+batch_size].cuda()
        else:
            inp, target = s[i:i+batch_size], a[i:i+batch_size]
#         print(inp.size(), target.size())
        oracle_loss = gen.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)

def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.
    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len x state_dim
        - neg_samples: neg_size x seq_len x state_dim
    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples.cpu(), neg_samples.cpu()), 0).type(torch.FloatTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target



