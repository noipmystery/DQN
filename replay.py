import random
import torch


class ReplayBuffer:
    """
    Experience replay buffer with limited size
    Sample with batches from experience replay
    """
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.cur = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.size:
            self.buffer[self.cur] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))
        self.cur = (self.cur + 1) % self.size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            state, action, reward, next_state, done = self.buffer[random.randint(0, len(self.buffer) - 1)]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (torch.tensor(states), torch.tensor(actions), torch.tensor(rewards),
                torch.tensor(next_states), torch.tensor(dones))
