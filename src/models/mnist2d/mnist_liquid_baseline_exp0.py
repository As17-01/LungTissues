import torch
import torch.nn.functional as f
from ncps.torch import LTC

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST2dLiquidBaselineExp0(BaseModel):
    def __init__(self):
        super(MNIST2dLiquidBaselineExp0, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.rnn1 = LTC(784, 128)
        self.rnn2 = LTC(128, 32)
        self.fc1 = torch.nn.Linear(32, 1)

    def forward(self, x):
        hidden1 = torch.zeros(x.shape[0], 128, device=get_default_device())
        hidden2 = torch.zeros(x.shape[0], 32, device=get_default_device())

        cur_x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))

        cur_x = x.view(x.shape[0], -1)

        cur_x = cur_x.unsqueeze(1)

        cur_x, hidden1 = self.rnn1(cur_x, hidden1)
        cur_x = f.relu(cur_x)
        cur_x, hidden2 = self.rnn2(cur_x, hidden2)
        cur_x = f.relu(cur_x)
        cur_x = f.sigmoid(self.fc1(cur_x))

        cur_x = cur_x.squeeze(1)

        return cur_x
