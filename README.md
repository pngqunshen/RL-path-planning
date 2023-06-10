# RL-path-planning

Path planning using reinforcement learning in a n x n grid environment:

![alt text](img/Maze.png?raw=true)

Starting position at (0, 0) (top left), and goal position at (n-1, n-1) (bottom right).

Current techniques
1. Monte-Carlo control without exploring starts
2. SARSA with an $\epsilon$-greedy behavior policy
3. Q-learning with an $\epsilon$-greedy behavior policy

## Monte-Carlo control without exploring starts
![alt text](img/Monte_carlo_without_es_pseudocode.png?raw=true)

## SARSA with an $\epsilon$-greedy behavior policy

Update rule:

$$Q\left(S_t,A_t\right)← Q\left(S_t,A_t\right)+\alpha\left[R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_t,A_t\right)\right]$$

![alt text](img/Sarsa_pseudocode.png?raw=true)

## Q-learning with an $\epsilon$-greedy behavior policy

Update rule:

$$Q\left(S_t,A_t\right)← Q\left(S_t,A_t\right)+\alpha\left[R_{t+1}+\gamma \text{max}_ {a'}Q\left(S_{t+1}, a'\right)-Q\left(S_t,A_t\right)\right]$$

![alt text](img/Q_learning_pseudocode.png?raw=true)

## $\epsilon$-greedy policy
$$\pi\left(a|s\right)
\begin{cases}
    1-\epsilon+\frac{\epsilon}{\left|A\left(s\right)\right|}, & \text{if}\ a=A^{\*}≜\text{argmax}_{a}Q\left(s,a\right) \\
    \frac{\epsilon}{\left|A\left(s\right)\right|}, & \text{if}\ a\neq A^{\*}
\end{cases}$$
