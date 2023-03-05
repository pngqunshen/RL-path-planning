# import statements
import numpy as np
import random
from RL import RL

class Monte_carlo_without_es(RL):
    def __init__(self, row, col, obstacle_pos = None, goal_pos = None, \
                 discount_rate = 0.9, epsilon = 0.3, seed = None):
        super().__init__(row, col, obstacle_pos, goal_pos, \
                       discount_rate, epsilon, seed)

        # initialise model
        self.Returns = {}

    ################################################################
    # find path
    ################################################################

    def generate_episode(self):
        i = 0
        j = 0
        ep = np.empty((0,4), int)
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1):
            pol = self.find_policy_to_use(i, j)
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
            ep = np.append(ep, np.array([[i, j, pol, self.reward[next_i, next_j]]], int), axis=0)
            i = next_i
            j = next_j
        return ep

    def update_path(self):
        ep = self.generate_episode()
        g = 0
        first_visit = np.zeros((self.row, self.col, self.num_actions))
        for i in range(len(ep)):
            S = ep[i, 0:2]
            A = ep[i, 2]
            if first_visit[S[0], S[1], A] == 0:
                first_visit[S[0], S[1], A] = i
        for i in range(len(ep) - 1, -1, -1):
            S = ep[i, 0:2]
            A = ep[i, 2]
            R = ep[i, 3]
            g = g * self.discount_rate + R
            if first_visit[S[0], S[1], A] == i:
                if ((S[0], S[1]), A) not in self.Returns:
                    self.Returns[((S[0], S[1]), A)] = np.empty((0), float)
                # if len(self.Returns[((S[0], S[1]), A)]) == self.row*self.col*self.num_actions:
                #     self.Returns[((S[0], S[1]), A)] = self.Returns[((S[0], S[1]), A)][1:]
                self.Returns[((S[0], S[1]), A)] = np.append(self.Returns[((S[0], S[1]), A)], g)
                self.q_value[S[0], S[1], A] = self.Returns[((S[0], S[1]), A)].mean()
                self.epsilon_greedy(S[0], S[1])