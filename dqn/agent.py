import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prioritized_memory import Memory

#BUFFER_SIZE = int(1e5)  # replay buffer size
#BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
#TAU = 1e-3              # for soft update of target parameters
#LR = 5e-4               # learning rate
#UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config_dict):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            config_dict (dict): Dictionary with configuration settings. Should
                have keys: seed, buffer_size, batch_size, gamma, loss_type,
                gradient_clip, fc1_units, fc2_units, lr, update_every, tau
        """
        # Check configs
        have_keys = ['seed', 'buffer_size', 'batch_size', 'gamma', 'loss_type',
                        'gradient_clip', 'fc1_units', 'fc2_units', 'lr',
                        'update_every', 'tau', 'use_dueling', 'per_a', 'per_beta', 
                        'use_per', 'use_ddqn']
        if not all([ x in config_dict for x in have_keys ]):
            raise ValueError("config_dict must have the following keys: %s" % ", ".join(have_keys))

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(config_dict['seed'])
        self.config = config_dict

        # Q-Network
        network = DuelingQNetwork if config_dict['use_dueling'] else QNetwork
        self.qnetwork_local = network(state_size, action_size, config_dict).to(device)
        self.qnetwork_target = network(state_size, action_size, config_dict).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])

        # Setup the loss function
        if config_dict['loss_type'] == 'huber':
            self.loss = nn.SmoothL1Loss() # similar to huber
        elif config_dict['loss_type'] == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("config_dict['loss_type'] should be 'huber' or 'mse'.")

        # Replay memory
        if config_dict['use_per']:
            self.memory = Memory(config_dict['buffer_size'], config_dict['per_a'], config_dict['per_beta'])
        else:
            self.memory = ReplayBuffer(action_size, config_dict['buffer_size'], config_dict['batch_size'], config_dict['batch_size'])

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def append_sample(self, state, action, reward, next_state, done):
        # Calcualte the priority or error
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

        p_state = torch.from_numpy(state).float().unsqueeze(0)
        p_nextstate = torch.from_numpy(next_state).float().unsqueeze(0)

        with torch.no_grad():
            local_argmax = self.qnetwork_local(p_nextstate).detach().argmax(dim=1).squeeze()
            Q_target_next = self.qnetwork_target(p_nextstate).detach().squeeze()[local_argmax]
            Q_target = reward + (self.config['gamma'] * Q_target_next * (1 - done))
            Q_expected = self.qnetwork_local(p_state).squeeze()[action]
            error = abs(Q_expected - Q_target)

        self.qnetwork_local.train()
        self.qnetwork_target.train()

        # Save experience in replay memory
        self.memory.add(error, state, action, reward, next_state, done)
        return

    def step(self, state, action, reward, next_state, done):
        # Adds the sample to memory with priority
        if self.config['use_per']:
            self.append_sample(state, action, reward, next_state, done)
            n = self.memory.tree.n_entries
        else:
            self.memory.add(state, action, reward, next_state, done)
            n = len(self.memory)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if n > self.config['batch_size']:
                if self.config['use_per']:
                    experiences, idxs, is_weights = self.memory.sample(self.config['batch_size'])
                    self.learn(experiences, idxs, is_weights)
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxs=None, is_weights=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            idx (list)
            is_weights (list)
        """
        gamma = self.config['gamma']

        states, actions, rewards, next_states, dones = experiences
        
        if self.config['use_ddqn']:
            # Do Double DQN instead of above
            # Select best action: Find max actions for next states based on the local_network
            local_argmax_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            # Evaluate that action: Use local_best_actions and target_network to get predicted value for next states
            Q_targets_next = self.qnetwork_target(next_states).gather(1,local_argmax_actions).detach()
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states (if done then ignore next state)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Update the priority
        if self.config['use_per']:
            errors = torch.abs(Q_expected - Q_targets).detach().numpy()
            for i in range(len(idxs)):
                self.memory.update(idxs[i], errors[i])

        # Compute loss
        if self.config['use_per']:
            loss = (torch.FloatTensor(is_weights) * self.loss(Q_expected, Q_targets)).mean()
        else:
            loss = self.loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip if there
        if self.config['gradient_clip']:
            for var in self.qnetwork_local.parameters():
                var.grad.data = torch.clamp(var.grad.data, -self.config['gradient_clip'], self.config['gradient_clip'])
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        tau = self.config['tau'] # interpolation paramater
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
