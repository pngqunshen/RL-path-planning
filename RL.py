# import statements
import numpy as np
import random

"""
General RL class to create an environment for learning
Child classes (Monte_Carlo_without_es, Sarsa, Q_learning) to extend
and implement the update_path() and generate_episode() methods

Usage:
1. Create child class
2. Call generate_path(n), where n is the number of iterations
"""
class RL():
    def __init__(self, row, col, obstacle_pos, goal_pos, \
                 discount_rate, epsilon, reward_shape, seed):
        # model params
        self.row = row # number of rows of grid
        self.col = col # number of columns of grid
        self.num_actions = 4
        """
        0 - up
        1 - right
        2 - down
        3 - left
        """

        # obstacles and goal
        self.obstacle_pos = obstacle_pos
        self.goal_pos = goal_pos

        # other params
        self.discount_rate = discount_rate
        self.epsilon = epsilon

        # initialise model
        # find reward without reward shaping
        self.reward = self.__initialise_reward(obstacle_pos, goal_pos, seed)
        # if reward_shape == None, no reward shaping
        if reward_shape == "manhattan": # perform man dist reward shaping
            self.reward = self.reward_shape_man_hat(self.reward)
        elif reward_shape == "apf": # perform apf reward shaping
            self.reward = self.reward_shape_apf(self.reward)
        self.policy = self.__initialise_policy()
        self.q_value = self.__initialise_q_value()

    ################################################################
    # initialise enviroment
    ################################################################

    def __initialise_reward(self, obstacle_pos, goal_pos, seed):
        reward = np.zeros((self.row,self.col))

        # check if seed is given
        if seed != None:
            random.seed(seed)
        # if obstacle not given, randomise 25% of tiles to be obstacles
        if obstacle_pos == None:
            new_obstacle_pos = [] # to set obstacle
            for i in range(int(0.25 * self.row * self.col)):
                while True:
                    # generate random spot to set as obstacle
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
        # use None as seed as the current seed value will not work
        return self.__initialise_reward(obstacle_pos, goal_pos, None)

    def __initialise_policy(self):
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
    
    
    def __initialise_q_value(self):
        # default q to 0 for all cell
        q_value = np.ones((self.row, self.col, self.num_actions))*0.1
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
    
    """
    Perform reward shaping to decrease the number of iterations required
    to train the model

    Based on manhattan distance from each cell to the goal
    """
    def reward_shape_man_hat(self, reward):
        res = reward.copy() # deep copy to prevent editing original
        goal_pos = np.where(reward == 1)
        others = np.where((reward != -1) == (reward != 1))
        """
        use manhattan distance to set reward of empty area
        reward for empty area = 1 - man dist / max possible man dist
        ensure that reward remains positive, and gets closer to 0
        the closer it is to the goal
        """
        for i in range(len(others[0])):
            res[others[0][i], others[1][i]] = \
                1-self.man_dist(goal_pos[0][0], goal_pos[1][0], \
                                others[0][i], others[1][i]) \
                                    / (self.row + self.col)
        return res

    """
    Perform reward shaping to decrease the number of iterations required
    to train the model

    Based on artificial potential field generated from each cell to the goal
    """
    def reward_shape_apf(self, reward):
        res = reward.copy() # deep copy to prevent editing original

        # find position index of important points
        goal_pos = np.where(reward == 1) # index of goal
        obs_pos = np.where(reward == -1) # index of obstacles
        others = np.where((reward != -1) == (reward != 1)) # index of other position

        # params for artificial potential field

        # maximum effective distance of goal, set to max manhattan distance
        # so entire map is in range
        MAX_ATT_DIST = (self.row + self.col)

        # maximum effective distance of repulsion, set to 2 so it only
        # affects the immediate 2 cells
        MAX_REP_DIST = 2

        # multiplier for attraction, chosen at 0.99 to ensure that the goal
        # will always have the highest reward
        ALPHA = 0.99

        # multiplier for repulsion
        BETA = 1

        for i in range(len(others[0])):
            # find manhattan distance for attraction
            att_r = self.man_dist(goal_pos[0][0], goal_pos[1][0], \
                                  others[0][i], others[1][i])
            # find attraction from goal
            att = ALPHA / att_r if att_r < MAX_ATT_DIST else 0
            potential = att # initialise potential to attraction first
            # add in repulsion from each obstacle
            for j in range(len(obs_pos[0])):
                # find manhattan distance for repulsion
                obs_r = self.man_dist(obs_pos[0][j], obs_pos[1][j], \
                                      others[0][i], others[1][i])
                # find repulsion from each obstacle
                rep = -BETA/(obs_r**2)*(1/obs_r-1/MAX_REP_DIST) if \
                    obs_r <= MAX_REP_DIST else \
                        0
                potential += rep # add the repulsion to potential
            # set the reward for this point
            res[others[0][i], others[1][i]] = potential
        return res
            
    ################################################################
    # helper
    ################################################################

    # calculate manhattan dist
    def man_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    # BFS to find if a path from start to goal is possible based on current reward, 
    # if not replant obstacles
    def bfs(self, reward):
        # case where origin is already obstacle
        if reward[0, 0] == -1:
            return False
        goal_pos = np.where(reward == 1) # index of goal
        obs_pos = np.where(reward == -1) # index of obstacles
        visited = np.zeros((self.row, self.col)) # visited nodes
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
        path = np.empty((0,2), int) # result path to return
        n = 0
        # loop as long as not in terminal state
        while not (self.reward[i,j] == -1 or self.reward[i,j] == 1) and \
              n < self.row * self.col:
            pol = self.q_value[i, j].argmax() # greedy policy, no exploration
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
            path = np.append(path, np.array([[i, j]]), axis = 0)
            # update to move into next cell
            i = next_i
            j = next_j
            n += 1
        # add in final step
        path = np.append(path, np.array([[i, j]]), axis = 0)
        return path
    
    
    
    ################################################################
    # find path
    ################################################################

    # generate a full episode based on current policy
    def generate_episode(self):
        pass # to be defined

    # do policy evaluation and improvement once
    def update_path(self, epsilon):
        pass # to be defined

    # determine the policy to use based on the probability of each action        
    def find_policy_to_use(self, i, j):
        # random number to generate policy
        pol = self.policy[i, j]
        rand = random.random()
        for k in range(len(pol)):
            if rand <= pol[k]:
                return k
            rand -= pol[k]
        return None # should not reach here
    
    # change the policy at state (i, j) to be epsilon greedy
    def epsilon_greedy(self, epsilon, i, j):
        # find total actions
        permitted_actions = []
        if i > 0: permitted_actions.append(0)
        if j < self.col - 1: permitted_actions.append(1)
        if i < self.row - 1: permitted_actions.append(2)
        if j > 0: permitted_actions.append(3)
        # set policy
        for act in permitted_actions:
            self.policy[i, j, act] = epsilon/len(permitted_actions)
        a_star = self.q_value[i, j].argmax() # best action based on value
        self.policy[i, j, a_star] = \
            1 - epsilon + epsilon/len(permitted_actions)

    """
    Main method to call after creating the object
    Program will break once a path to goal is found, and return the
    number of iterations taken

    n - maximum number of times to iterate
    """                   
    def generate_path(self, n):
        for i in range(n):
            # calculate decayed epsilon, used if epsilon not defined
            # epsilon decreases linearly over the entire episode, starting at 1
            # and ending at p_end (defined below)
            p_end = 0.1
            epsilon = (1 - i/(n-1)) * (1- p_end) + p_end

            # generate path and check if converge
            # convergence is defined as a path to the goal, which may not
            # necessarily be optimal
            if self.epsilon == None: # epsilon not defined, use decayed epsilon
                self.update_path(epsilon)
            else:
                self.update_path(self.epsilon) # epsilon defined
            path = self.get_best_path()
            if path[-1,0]==self.goal_pos[0] and path[-1,1]==self.goal_pos[1]:
                break
        return i, self.get_best_path()
    
    ################################################################
    # print map
    ################################################################

    """
    To print the environment for easier visualisation
    X - Obstacles
    G - Goal
    """
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
    

    """
    To print the environment with best path currently
    for easier visualisation
    X - Obstacles
    G - Goal
    O - path
    """
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