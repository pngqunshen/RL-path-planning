# import statements
import numpy as np
import random
from RL import RL

class Q_learning(RL):
    def __init__(self, row, col, obstacle_pos = None, goal_pos = None, \
                 discount_rate = 0.9, epsilon = None, reward_shape = None, seed = None):
        super().__init__(row, col, obstacle_pos, goal_pos, \
                       discount_rate, epsilon, reward_shape, seed)

    ################################################################
    # find path
    ################################################################

    def generate_episode(self):
        i = 0
        j = 0
        ep = np.empty((0,7), int) # contains information from each episode
        pol = self.find_policy_to_use(i, j)
        # (state_i, state_j, action, reward, next_state_i, next_state_j, next_action)
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1):
            next_i = i
            next_j = j
            # find next cell
            if pol == 0: # move up
                next_i = max(i-1, 0)
            elif pol == 1: # move right
                next_j = min(j+1, self.col-1)
            elif pol == 2: # move down
                next_i = min(i+1, self.row-1)
            else: # move left
                next_j = max(j-1, 0)
            next_pol = self.find_policy_to_use(next_i, next_j)
            # update episode with current state
            ep = np.append(ep, np.array([[i, j, pol, self.reward[next_i, next_j], \
                                          next_i, next_j, next_pol]], int), axis=0)
            i = next_i
            j = next_j
            pol = next_pol
        return ep

    def update_path(self, epsilon):
        # initialise episode
        ep = self.generate_episode()
        for i in range(len(ep)): # loop through entire episode in order
            S = ep[i, 0:2] # current state
            A = ep[i, 2] # action
            R = ep[i, 3] # reward
            SN = ep[i, 4:6] # next state
            AN = ep[i, 6] # next action
            learning_rate = 1 / i if i != 0 else 1 # learning rate alpha = 1/t
            self.q_value[S[0], S[1], A] += \
                (learning_rate*(R+self.discount_rate*self.q_value[SN[0], SN[1]].max() \
                             -self.q_value[S[0], S[1], A]))
            self.epsilon_greedy(epsilon, S[0], S[1])