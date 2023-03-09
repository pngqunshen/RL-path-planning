# import statements
import numpy as np
import random
from RL import RL

class Sarsa(RL):
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
        i = 0
        j = 0
        pol = self.find_policy_to_use(i, j)
        t = 0
        while self.reward[i, j] != 1 and self.reward[i, j] != -1:
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
            self.epsilon_greedy(epsilon, next_i, next_j)
            next_pol = self.find_policy_to_use(next_i, next_j)
            learning_rate = 1 / t if t != 0 else 1 # learning rate alpha = 1/t
            self.q_value[i, j, pol] += \
                (learning_rate*(self.reward[i, j]+self.discount_rate*self.q_value[next_i, next_j, next_pol] \
                             -self.q_value[i, j, pol]))
            i = next_i
            j = next_j
            pol = next_pol
            t += 1