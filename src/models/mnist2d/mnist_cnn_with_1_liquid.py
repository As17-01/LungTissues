import torch
import torch.nn.functional as f
from ncps.torch import LTC

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST2dCNNWith1Liquid(BaseModel):
    def __init__(self):
        super(MNIST2dCNNWith1Liquid, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, 120)
        self.fc1 = torch.nn.Linear(120, 84)
        self.fc2 = torch.nn.Linear(84, 1)

    def forward(self, x):
        hidden1 = torch.zeros(x.shape[0], 120, device=get_default_device())

        cur_x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
        cur_x = cur_x.view(x.shape[0], -1)

        cur_x = cur_x.unsqueeze(1)
        cur_x, hidden1 = self.rnn1(cur_x, hidden1)
        cur_x = cur_x.squeeze(1)

        cur_x = f.relu(cur_x)
        cur_x = f.relu(self.fc1(cur_x))
        cur_x = f.sigmoid(self.fc2(cur_x))
        return cur_x
