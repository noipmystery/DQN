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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.buffer)

    def transform(self, lazy_frame):
        state = torch.from_numpy(lazy_frame.__array__()[None] / 255).float()
        return state.to(self.device)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.size:
            self.buffer[self.cur] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))
        self.cur = (self.cur + 1) % self.size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            frame, action, reward, next_frame, done = self.buffer[random.randint(0, len(self.buffer) - 1)]
            state = self.transform(frame)
            next_state = self.transform(next_frame)
            state = torch.squeeze(state, 0)
            next_state = torch.squeeze(next_state, 0)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (torch.stack(states).to(self.device), torch.tensor(actions).to(self.device), torch.tensor(rewards).to(self.device),
                torch.stack(next_states).to(self.device), torch.tensor(dones).to(self.device))
