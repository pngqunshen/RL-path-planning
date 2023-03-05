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

$Q(S_t,A_t)← Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1}, A_{t+1})-Q(S_t,A_t)]$

![alt text](img/Sarsa_pseudocode.png?raw=true)

## Q-learning with an $\epsilon$-greedy behavior policy

Update rule:

$Q(S_t,A_t)← Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \text{max}_{a'}Q(S_{t+1}, a')-Q(S_t,A_t)]$

![alt text](img/Q_learning_pseudocode.png?raw=true)

## $\epsilon$-greedy policy
$\pi(a|s)=
\begin{cases}
    1-\epsilon+\frac{\epsilon}{|A(s)|}, & \text{if}\ a=A^*≜\text{argmax}_aQ(s,a) \\
    \frac{\epsilon}{|A(s)|}, & \text{if}\ a\neq A^*
\end{cases}$