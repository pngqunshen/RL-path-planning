# import statements
import numpy as np
import random
from RL import RL

class Monte_carlo_without_es(RL):
    def __init__(self, row, col, obstacle_pos = None, goal_pos = None, \
                 discount_rate = 0.9, epsilon = None, reward_shape = None, seed = None):
        super().__init__(row, col, obstacle_pos, goal_pos, \
                       discount_rate, epsilon, reward_shape, seed)

        # initialise model
        self.Returns = {}

    ################################################################
    # find path
    ################################################################

    def generate_episode(self):
        i = 0
        j = 0
        ep = np.empty((0,4), int) # contains information from each episode
        # (state_i, state_j, action, reward)
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1):
            pol = self.find_policy_to_use(i, j)
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
            # update episode with current state
            ep = np.append(ep, np.array([[i, j, pol, self.reward[next_i, next_j]]], int), axis=0)
            i = next_i
            j = next_j
        return ep

    def update_path(self, epsilon):
        # initialise episode and g value
        ep = self.generate_episode()
        g = 0
        # keeps track of which time step is the first time a action state pair is visited
        # initialise each state and action pair to -1, loop through in order
        # and change state action pair value to 1 in first visit
        first_visit = np.ones((self.row, self.col, self.num_actions)) * -1
        for i in range(len(ep)): # loop through entire episode in order
            S = ep[i, 0:2]
            A = ep[i, 2]
            # -1 means not visited yet, this time step is first visit
            if first_visit[S[0], S[1], A] == -1:
                first_visit[S[0], S[1], A] = i
        for i in range(len(ep) - 1, -1, -1): # loop through entire episode in reverse
            S = ep[i, 0:2]
            A = ep[i, 2]
            R = ep[i, 3]
            g = g * self.discount_rate + R
            # following action done only if this is first visit, else only update g above
            if first_visit[S[0], S[1], A] == i:
                if ((S[0], S[1]), A) not in self.Returns:
                    self.Returns[((S[0], S[1]), A)] = np.empty((0), float)
                # keep a rolling average window of size == number of state action pair
                if len(self.Returns[((S[0], S[1]), A)]) == self.row*self.col*self.num_actions:
                    self.Returns[((S[0], S[1]), A)] = self.Returns[((S[0], S[1]), A)][1:]
                self.Returns[((S[0], S[1]), A)] = np.append(self.Returns[((S[0], S[1]), A)], g)
                self.q_value[S[0], S[1], A] = self.Returns[((S[0], S[1]), A)].mean()
                self.epsilon_greedy(epsilon, S[0], S[1])