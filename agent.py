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
    def __init__(self, in_channels, num_actions, c, lr, alpha, gamma, epsilon, replay_size):
        self.num_actions = num_actions
        self.replay = ReplayBuffer(replay_size)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.c = c
        self.gamma = gamma
        self.q_network = DQN(in_channels, num_actions).to(self.device)
        self.target_network = DQN(in_channels, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr, eps=epsilon, alpha=alpha)
        # self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-3)

    def greedy(self, state, epsilon):
        """
        Take actions with state under epsilon-greedy policy
        """
        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            q_values = self.q_network(state).detach().cpu().numpy()
            action = np.argmax(q_values)
            del q_values
        return action

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        y(state) = reward if done
                 = reward + gamma * max_a target(next_state, a)
        loss = (y(state) - q_network(state, action)) ^ 2
        """
        tmp = self.q_network(states)
        rewards = rewards.to(self.device)
        q_values = tmp[range(states.shape[0]), actions.long()]
        default = rewards + self.gamma * self.target_network(next_states).max(dim=1)[0]
        target = torch.where(dones.to(self.device), rewards, default).to(self.device).detach()
        return F.mse_loss(target, q_values)

    def reset(self):
        """
        Reset target_network from q_network every C steps
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self, batch_size):
        if batch_size < len(self.replay):
            states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
            loss = self.calculate_loss(states, actions, rewards, next_states, dones)
            # print(f"loss : {loss}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0
