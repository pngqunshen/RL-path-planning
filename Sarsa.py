# import statements
import numpy as np
import random

class Sarsa():
    def __init__(self, row, col, obstacle_pos = None, goal_pos = None, \
                 discount_rate = 0.9, epsilon = 0.1, seed = 1):
        # model params
        self.row = row # number of rows of grid
        self.col = col # number of columns of grid
        self.num_actions = 4

        # initialise model
        self.reward = self.initialise_reward(obstacle_pos, goal_pos, seed)
        self.policy = self.initialise_policy()
        self.q_value = self.initialise_q_value()

        # other params
        self.discount_rate = discount_rate
        self.epsilon = epsilon

    def initialise_reward(self, obstacle_pos, goal_pos, seed):
        reward = np.ones((self.row,self.col))*-0.04

        # if obstacle not given, randomise 25% of tiles to be obstacles
        random.seed(seed)
        if obstacle_pos == None:
            for i in range(int(0.25 * self.row * self.col)):
                while True:
                    rand_i = random.randint(0, self.row-1)
                    rand_j = random.randint(0, self.col-1)
                    if reward[rand_i, rand_j] != -1 and reward[rand_i, rand_j] != 1: 
                        reward[rand_i, rand_j] = -1
                        break
        else: # use given obstacles
            for (i, j) in obstacle_pos:
                reward[i,j] = -1
        # if goal not given, default to lower right
        if goal_pos == None:
            reward[self.row-1, self.col-1] = 1
        else:
            reward[goal_pos[0], goal_pos[1]] = 1
        return reward

    def initialise_policy(self):
        pol = np.ones((self.row, self.col, self.num_actions))*(1.0/self.num_actions)
        # remove probabilty of going into the wall
        pol[0,:] = np.repeat([[0, 1.0/3, 1.0/3, 1.0/3]], self.col, axis=0)
        pol[self.row-1,:] = np.repeat([[1.0/3, 1.0/3, 0, 1.0/3]], self.col, axis=0)
        pol[:,0] = np.repeat([[1.0/3, 1.0/3, 1.0/3, 0]], self.col, axis=0)
        pol[:,self.col-1] = np.repeat([[1.0/3, 0, 1.0/3, 1.0/3]], self.col, axis=0)
        # update probability of corner
        pol[0,0] = [0, 1.0/2, 1.0/2, 0]
        pol[0,self.col-1] = [0, 0, 1.0/2, 1.0/2]
        pol[self.row-1,0] = [1.0/2, 1.0/2, 0, 0]
        pol[self.row-1,self.col-1] = [1.0/2, 0, 0, 1.0/2]
        return pol
    
    def initialise_q_value(self):
        q_value = np.zeros((self.row, self.col, self.num_actions))
        # set q_value of going into the wall to low number
        q_value[0,:,0] = -2147483648
        q_value[:,self.col-1,1] = -2147483648
        q_value[self.row-1,:,2] = -2147483648
        q_value[:,0,3] = -2147483648
        return q_value

    def generate_episode(self):
        i = 0
        j = 0
        ep = np.empty((0,7), int)
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
            next_pol = self.find_policy_to_use(next_i, next_j)
            ep = np.append(ep, np.array([[i, j, pol, self.reward[next_i, next_j], \
                                          next_i, next_j, next_pol]], int), axis=0)
            i = next_i
            j = next_j
        return ep
            
    def find_policy_to_use(self, i, j):
        # random number to generate policy
        pol = self.policy[i, j]
        rand = random.random()
        for k in range(len(pol)):
            if rand <= pol[k]:
                return k
            rand -= pol[k]
        return None # should not reach here
    
    def update_path(self):
        ep = self.generate_episode()
        for i in range(len(ep)):
            S = ep[i, 0:2]
            A = ep[i, 2]
            R = ep[i, 3]
            SN = ep[i, 4:6]
            AN = ep[i, 6]
            alpha = 1 / i if i != 0 else 1
            self.q_value[S[0], S[1], A] += \
                (alpha*(R+self.discount_rate*self.q_value[SN[0], SN[1], AN] \
                             -self.q_value[S[0], S[1], A]))
            self.epsilon_greedy(S[0], S[1])
            
    def epsilon_greedy(self, i, j):
        # find total actions
        permitted_actions = []
        if i > 0: permitted_actions.append(0)
        if j < self.col - 1: permitted_actions.append(1)
        if i < self.row - 1: permitted_actions.append(2)
        if j > 0: permitted_actions.append(3)
        # set policy
        for act in permitted_actions:
            self.policy[i, j, act] = self.epsilon/len(permitted_actions)
        a_star = self.q_value[i, j].argmax()
        self.policy[i, j, a_star] = \
            1 - self.epsilon + self.epsilon/len(permitted_actions)
    
    def generate_path(self):
        ep = self.generate_episode()
        while True:
            self.update_path()
            new_ep = self.generate_episode()
            if np.array_equal(ep, new_ep) and ep[-1][3] == 1:
                break
            ep = new_ep
        return ep