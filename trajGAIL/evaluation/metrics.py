import numpy as np
import difflib
from matplotlib import pyplot as plt

'''
    Amount of route distance shared & Percentage of testing paths with above 90% matching.
    input: generated trajectories in states and actions, expert trajectories
    output: Amount of route distance shared & Percentage of testing paths with above 90% matching
'''
def route_sharing_average_percentage(gen_ts, gen_ta, exp_traj):
    test_dict = {}
    for traj in exp_traj:
        if traj[0][0] not in test_dict:
            test_dict[traj[0][0]] = []
        temp = []
        for step in traj:
            temp.append(step[-2])
        test_dict[traj[0][0]].append(temp)
        
    tg_dict = {}
    for traj_st, traj_a in zip(gen_ts, gen_ta):
        if int(traj_st[0][0].item()) not in tg_dict:
            tg_dict[int(traj_st[0][0].item())] = []
        temp = []
        for a in traj_a:
            temp.append(a.item())
        tg_dict[int(traj_st[0][0].item())].append(temp)
    
    sim = []
    for key, trajs in test_dict.items():
        if key in tg_dict:
            similarity = []
            for etraj in trajs:
                for gtraj in tg_dict[key]:
                    similarity.append(difflib.SequenceMatcher(None, etraj, gtraj).ratio())
            if len(similarity) > 0:
                sim.append(max(similarity))
    print('RS: {}%; >90: {}%'.format(np.average(sim)*100., sum([x>=0.9 for x in sim])/len(sim)*100))
    return np.average(sim), sum([x>=0.9 for x in sim])/len(exp_traj)

'''Visal comparision of length distribution difference'''
def plot_length_distribution(gen_traj, gen_bin, exp_traj, exp_bin, path):
    '''Get the length of gen_traj'''
    tg_length = []
    for traj in gen_traj:
        count = 0
        for a in traj:
            if a == 9:
                tg_length.append(count)
                count = 0
                break
            count += 1
    plt.figure()
    plt.hist([tg_length], bins=gen_bin, alpha=0.5, label='generated')
    plt.hist([len(traj) for traj in exp_traj], bins=10, alpha=0.5, label='expert')
    plt.legend(loc='upper right')
#     plt.show()
    plt.savefig(path)

