import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Actor class for the agent, which will be used to take actions based on the current state.
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution
    
# Critic class for the agent, which will be used to evaluate the current state.
class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.layer3(x)
        return value