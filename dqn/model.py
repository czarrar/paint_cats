import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, config_dict):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config_dict (dict): Config (fc1_units and fc2_units)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(config_dict['seed'])
        self.fc1 = nn.Linear(state_size, config_dict['fc1_units'])
        self.fc2 = nn.Linear(config_dict['fc1_units'], config_dict['fc2_units'])
        self.fc3 = nn.Linear(config_dict['fc2_units'], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, config_dict):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config_dict (dict): Config (fc1_units and fc2_units)
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(config_dict['seed'])
        self.fc1 = nn.Linear(state_size, config_dict['fc1_units'])
        self.fc2 = nn.Linear(config_dict['fc1_units'], config_dict['fc2_units'])
        self.advantages = nn.Linear(config_dict['fc2_units'], action_size)
        self.values = nn.Linear(config_dict['fc2_units'], 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.values(x)
        a = self.advantages(x)
        return v + a - a.mean()
