# import statements
import numpy as np
import random

class RL():
    def __init__(self, row, col, obstacle_pos, goal_pos, \
                 discount_rate, epsilon, seed):
        # model params
        self.row = row # number of rows of grid
        self.col = col # number of columns of grid
        self.num_actions = 4

        # obstacles and goal
        self.obstacle_pos = obstacle_pos
        self.goal_pos = goal_pos

        # other params
        self.discount_rate = discount_rate
        self.epsilon = epsilon

        # initialise model
        self.reward = self.reward_shape(self.initialise_reward(obstacle_pos, goal_pos, seed))
        self.policy = self.initialise_policy()
        self.q_value = self.initialise_q_value()

    ################################################################
    # initialise map
    ################################################################

    def initialise_reward(self, obstacle_pos, goal_pos, seed):
        reward = np.zeros((self.row,self.col))

        # check if seed is given
        if seed != None:
            random.seed(seed)
        # if obstacle not given, randomise 25% of tiles to be obstacles
        if obstacle_pos == None:
            new_obstacle_pos = [] # to set obstacle
            for i in range(int(0.25 * self.row * self.col + 1)):
                while True:
                    rand_i = random.randint(0, self.row-1)
                    rand_j = random.randint(0, self.col-1)
                    # ensure obstacle placed only in empty area
                    if reward[rand_i, rand_j] != -1 and reward[rand_i, rand_j] != 1: 
                        reward[rand_i, rand_j] = -1
                        new_obstacle_pos.append((rand_i, rand_j))
                        break
            self.obstacle_pos = new_obstacle_pos
        else: # use given obstacles
            for (i, j) in obstacle_pos:
                reward[i,j] = -1
        # if goal not given, default to lower right
        if goal_pos == None:
            self.goal_pos = (self.row-1, self.col-1)
            reward[self.row-1, self.col-1] = 1
        else:
            reward[goal_pos[0], goal_pos[1]] = 1
        # use bfs to check if a path is even possible, if not redo
        if self.bfs(reward):
            return reward
        return self.initialise_reward(obstacle_pos, goal_pos, None)

    def initialise_policy(self):
        # initially, equal probability of going in any of the legal direction
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
        # default q to 0 for all cell
        q_value = np.zeros((self.row, self.col, self.num_actions))
        # set q_value of going into the wall to low number, so it will
        # never be considered
        q_value[0,:,0] = -2147483648
        q_value[:,self.col-1,1] = -2147483648
        q_value[self.row-1,:,2] = -2147483648
        q_value[:,0,3] = -2147483648
        # set value at obstacle to 0
        for obs in self.obstacle_pos:
            q_value[obs[0], obs[1]] = np.zeros((self.num_actions))
        return q_value
    
    def reward_shape(self, reward):
        res = reward.copy() # deep copy to prevent editing original
        goal_pos = np.where(reward == 1)
        others = np.where((reward != -1) == (reward != 1))
        # use manhattan distance to set reward of empty area
        # reward for empty area = 1 - man dist / max possible man dist
        for i in range(len(others[0])):
            res[others[0][i], others[1][i]] = \
                1-self.man_dist(goal_pos[0][0], goal_pos[1][0], \
                                others[0][i], others[1][i]) \
                                    / (self.row + self.col)
        return res
            
    ################################################################
    # helper
    ################################################################

    # calculate manhattan dist
    def man_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    # BFS to find if a path is possible, if not replant obstacles
    def bfs(self, reward):
        # case where origin is already obstacle
        if reward[0, 0] == -1:
            return False
        goal_pos = np.where(reward == 1)
        obs_pos = np.where(reward == -1)
        visited = np.zeros((self.row, self.col))
        # prevent visits to obstacle
        for i in range(len(obs_pos[0])):
            visited[obs_pos[0][i], obs_pos[1][i]] = 1
        q = [(0, 0)] # queue to keep track of visiting nodes
        while len(q) > 0:
            curr = q.pop(0)
            visited[curr[0], curr[1]] = 1 # visit curr node
            # if reach goal, path is possible, return true
            if curr[0] == goal_pos[0][0] and curr[1] == goal_pos[1][0]:
                return True
            # add neighbour into queue
            if curr[0] > 0 and visited[curr[0]-1, curr[1]] == 0:
                q.append((curr[0]-1, curr[1]))
            if curr[1] < self.col - 1 and visited[curr[0], curr[1]+1] == 0:
                q.append((curr[0], curr[1]+1))
            if curr[0] < self.row - 1 and visited[curr[0]+1, curr[1]] == 0:
                q.append((curr[0]+1, curr[1]))
            if curr[1] > 0 and visited[curr[0], curr[1]-1] == 0:
                q.append((curr[0], curr[1]-1))
        # finish BFS, goal not found, return false
        return False
    
    # get best path based on greedy policy
    def get_best_path(self):
        i = 0
        j = 0
        path = np.empty((0,2), int)
        n = 0
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1) and \
              n < self.row * self.col:
            pol = self.q_value[i, j].argmax()
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
            path = np.append(path, np.array([[i, j]]), axis = 0)
            i = next_i
            j = next_j
            n += 1
        path = np.append(path, np.array([[i, j]]), axis = 0)
        return path
    
    ################################################################
    # find path
    ################################################################

    def generate_episode(self):
        pass # to be defined

    def update_path(self):
        pass # to be defined
            
    def find_policy_to_use(self, i, j):
        # random number to generate policy
        pol = self.policy[i, j]
        rand = random.random()
        for k in range(len(pol)):
            if rand <= pol[k]:
                return k
            rand -= pol[k]
        return None # should not reach here
    
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
                        
    def generate_path(self, n):
        # path = self.get_best_path()
        # j = 0
        for i in range(n):
            self.update_path()
            # new_path = self.get_best_path()
            # if np.array_equal(path, new_path) and \
            #     self.reward[new_path[-1][0], new_path[-1][1]] == 1:
            #     j += 1
            # else:
            #     j = 0
            # path = new_path
            # if (j == 100):
            #     break
        return self.get_path_map()
    
    ################################################################
    # print map
    ################################################################

    def get_map(self):
        res = "=" * (self.col * 3 + 2)
        for i in self.reward:
            line = "\n|"
            for j in i:
                if j == -1:
                    line += " X "
                elif j == 1:
                    line += " G "
                else:
                    line += "   "
            line += "|"
            res += line
        res += ("\n" + "=" * (self.col * 3 + 2))
        return res
    
    def get_path_map(self):
        path = self.get_best_path()
        path_map = []
        for i in self.reward:
            tmp = []
            for j in i:
                if j == -1:
                    tmp.append(" X ")
                elif j == 1:
                    tmp.append(" G ")
                else:
                    tmp.append("   ")
            path_map.append(tmp)
        for i in path:
            if path_map[i[0]][i[1]] == " G ":
                break
            if path_map[i[0]][i[1]] == " X ":
                break
            path_map[i[0]][i[1]] = " O "
        res = "=" * (self.col * 3 + 2)
        for i in path_map:
            line = "\n|"
            for j in i:
                line += j
            line += "|"
            res += line
        res += ("\n" + "=" * (self.col * 3 + 2))
        return res