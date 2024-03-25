from environment import make_env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import agent
import math
import torch

batch_size = 32
learning_rate = 2.5e-4
gamma = 0.99
epsilon_begin = 1.0
epsilon_end = 0.1
epsilon_decay = 30000
epsilon_min = 0.01
alpha = 0.95
memory_size = 1000000
replay_start_size = 5000
total_frame = 2000000
update = 10000
print_interval = 1000


def epsilon(cur):
    return epsilon_end + (epsilon_begin - epsilon_end) * math.exp(-1.0 * cur / epsilon_decay)


if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    agent = agent.Agent(in_channels=1, num_actions=env.action_space.n, c=update,
                        lr=learning_rate, alpha=alpha, gamma=gamma, epsilon=epsilon_min, replay_size=memory_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = env.reset()[0]
    state = torch.from_numpy(state[None] / 255).float()
    state = state.to(device)
    total_reward = 0
    Loss = []
    Reward = []
    episodes = 0
    writer = SummaryWriter(log_dir='./logs')
    for _ in range(total_frame):
        eps = epsilon(_)
        action = agent.greedy(state, epsilon=eps)
        next_state, reward, done, info, tmp = env.step(action)
        # print('frame {}, done: {}'.format(_, done))
        next_state = torch.from_numpy(next_state[None] / 255).float().to(device)
        agent.replay.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        loss = 0

        if len(agent.replay) > replay_start_size:
            # print('learning')
            loss = agent.learn(batch_size=batch_size)
            # print(loss)
            Loss.append(loss)

        if _ % agent.c == 0:
            agent.reset()

        if _ % print_interval == 0:
            print('frame : {}, loss : {:.4f}, reward : {}'.format(_, loss, np.mean(Reward[-10:])))
            writer.add_scalar('loss', loss, _)
            writer.add_scalar('reward', np.mean(Reward[-10:]), _)

        if done:
            episodes += 1
            Reward.append(total_reward)
            print('episode {}: total reward {}'.format(episodes, total_reward))
            state = env.reset()[0]
            state = torch.from_numpy(state[None] / 255).float()
            state = state.to(device)
            total_reward = 0

    writer.close()
