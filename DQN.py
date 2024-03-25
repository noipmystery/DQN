import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    """
    Deep Q-learning network
    The same architecture from the paper https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    Arguments:
        in_channels: channels of the input image
        num_actions: total number of actions
    """
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        # print('forward: shape = {}'.format(x.shape))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
