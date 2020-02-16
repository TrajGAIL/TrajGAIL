import random
import torch
import os

STATE_DIM = 125

def judging_action(x, y, t, nx, ny, nt):
    if x == 0 and y == 0:
        return 9 # stop
    if nx == 0 and ny == 0:
        return 9 # stop
    pluson = 0
    if nt != t:
        pluson = 10
    if x == nx and ny > y:
        return pluson + 0 # 90
    if x < nx and ny > y:
        return pluson + 1 # 45
    if x < nx and ny == y:
        return pluson + 2 # 0
    if x < nx and ny < y:
        return pluson + 3 # -45
    if x == nx and ny < y:
        return pluson + 4 # -90
    if x > nx and ny < y:
        return pluson + 5 # -135
    if x > nx and ny == y:
        return pluson + 6 # 180
    if x > nx and ny > y:
        return pluson + 7 # 135
    if x == nx and y == ny:
        return pluson + 8 # stay

def decide_next_state(dirc, current_state):
    if current_state [0] == 0:
        return [0., 0., 0., 0.]
    if int(dirc/10) == 1:
        new_time = (current_state[2]+1)%288
    else:
        new_time = current_state[2]
    dirc = dirc%10
    
    if dirc == 0:
        new_step = [current_state[0], current_state[1]+1, new_time, current_state[-1]]
        return new_step
    if dirc == 1: 
        new_step = [current_state[0]+1, current_state[1]+1, new_time, current_state[-1]] 
        return new_step
    if dirc == 2:
        new_step = [current_state[0]+1, current_state[1], new_time, current_state[-1]]
        return new_step
    if dirc == 3: 
        new_step = [current_state[0]+1, current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 4:
        new_step = [current_state[0], current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 5: 
        new_step = [current_state[0]-1, current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 6:
        new_step = [current_state[0]-1, current_state[1], new_time, current_state[-1]]
        return new_step
    if dirc == 7:
        new_step = [current_state[0]-1, current_state[1]+1, new_time, current_state[-1]]
        return new_step
    if dirc == 8:
        new_step = [current_state[0], current_state[1], new_time, current_state[-1]]
        return new_step
    else:
        return [0., 0., 0., 0.]

def state_action_separation(trajectories):
    state = []
    action = []
    for traj in trajectories:
        temp_state_traj = []
        temp_action_traj = []
        for step in traj:
            temp_state_traj.append(step[:125])
            temp_action_traj.append(step[-1])
        state.append(temp_state_traj)
        action.append(temp_action_traj)
    return state, action
