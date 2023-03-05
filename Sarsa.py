# import statements
import numpy as np
import random
from RL import RL

class Sarsa(RL):
    def __init__(self, row, col, obstacle_pos = None, goal_pos = None, \
                 discount_rate = 0.9, epsilon = 0.1, seed = 1):
        super().__init__(row, col, obstacle_pos, goal_pos, \
                       discount_rate, epsilon, seed)

    ################################################################
    # find path
    ################################################################

    def generate_episode(self):
        i = 0
        j = 0
        ep = np.empty((0,7), int)
        pol = self.find_policy_to_use(i, j)
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1):
            next_i = i
            next_j = j
            if pol == 0:
                next_i = max(i-1, 0)
            elif pol == 1:
                next_j = min(j+1, self.col-1)
            elif pol == 2:
                next_i = min(i+1, self.row-1)
            else:
                next_j = max(j-1, 0)
            next_pol = self.find_policy_to_use(next_i, next_j)
            ep = np.append(ep, np.array([[i, j, pol, self.reward[next_i, next_j], \
                                          next_i, next_j, next_pol]], int), axis=0)
            i = next_i
            j = next_j
            pol = next_pol
        return ep

    def update_path(self):
        ep = self.generate_episode()
        for i in range(len(ep)):
            S = ep[i, 0:2]
            A = ep[i, 2]
            R = ep[i, 3]
            SN = ep[i, 4:6]
            AN = ep[i, 6]
            learning_rate = 1 / i if i != 0 else 1
            self.q_value[S[0], S[1], A] += \
                (learning_rate*(R+self.discount_rate*self.q_value[SN[0], SN[1], AN] \
                             -self.q_value[S[0], S[1], A]))
            self.epsilon_greedy(S[0], S[1])