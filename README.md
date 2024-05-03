# woRdLe : A Deep Reinforcement Learning Approach

Team Members:
Harsh Shah, Anwit Damle, Ujjwal Agarwal

Requirements: 

pip install numpy
pip install torch


Environment:
State: A 78-dimensional vector encodes each letter's correctness and position relative to the target word.
Action: Selecting a word from a dictionary, with masking to prevent repeated guesses.
Reward System
Correct Word: +10 reward for guessing the target word correctly. 
Attempts Exhausted: -10 penalty for exhausting attempts without guessing correctly.
Midway Progress: +1 reward for each correctly positioned letter in the guessed word.

Agents Applied :

1. Double Deep Q-Network:
DDQN employs two separate neural networks, known as the online network and the target network, to decouple the action selection from the value estimation. By periodically updating the target network with the parameters of the online network, DDQN stabilizes the learning process and improves performance, particularly in environments with large action spaces or complex dynamics.

2. Advantage Actor Critic:
In A2C, an actor network learns a policy to select actions, while a critic network estimates the value function to evaluate these actions. The advantage of A2C lies in its ability to update both the policy and the value function simultaneously, using the advantage function to guide learning. This approach leads to more stable training and faster convergence compared to traditional policy gradient methods.
The advantage function, A(s,a), measures the advantage of taking action a in state s​ over the expected value of the state under the current policy.
A(s,a) = Q(s,a) - V(s)

Github Link: [repo](https://github.com/harsh788/woRdLe)

Used resources:

https://www.nytimes.com/games/wordle/index.html

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/