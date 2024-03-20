from DQN import DQN
from replay import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class Agent:
    """
    Agent with q-network and target network, with epsilon greedy
    Optimizer: RMSProp
    """
    def __init__(self, in_channels, num_actions, lr, alpha, gamma, epsilon, replay_size):
        self.num_actions = num_actions
        self.replay = ReplayBuffer(replay_size)
        self.gamma = gamma
        self.q_network = DQN(in_channels, num_actions)
        self.target_network = DQN(in_channels, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr, eps=epsilon, alpha=alpha)

    def greedy(self, state, epsilon):
        """
        Take actions with state under epsilon-greedy policy
        """
        q_values = self.q_network(state).numpy()
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(q_values)

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        y(state) = reward if done
                 = reward + gamma * max_a target(next_state, a)
        loss = (y(state) - q_network(state, action)) ^ 2
        """
        action = F.one_hot(actions, self.num_actions)
        tmp = self.q_network(states)
        q_values = torch.sum(torch.multiply(action, tmp), dim=0)
        default = rewards + self.gamma * torch.max(self.target_network(next_states), dim=1)
        target = torch.where(dones, rewards, default)
        return F.mse_loss(target, q_values)

    def reset(self):
        """
        Reset target_network from q_network every C steps
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self, batch_size):
        if batch_size > len(self.replay):
            states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
            loss = self.calculate_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0
