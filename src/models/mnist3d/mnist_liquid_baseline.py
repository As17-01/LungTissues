import torch
import torch.nn.functional as f
from ncps.torch import LTC

from src.models.base import BaseModel
from src.utils import get_default_device


class MNISTLiquidBaseline(BaseModel):
    def __init__(self):
        super(MNISTLiquidBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.rnn1 = LTC(256, 128)
        self.rnn2 = LTC(128, 64)
        self.rnn3 = LTC(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        hidden1 = torch.zeros(x.shape[0], 128, device=get_default_device())
        hidden2 = torch.zeros(x.shape[0], 64, device=get_default_device())
        hidden3 = torch.zeros(x.shape[0], 32, device=get_default_device())

        cur_x = x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))

        cur_x = cur_x.unsqueeze(1)

        cur_x, hidden1 = self.rnn1(cur_x, hidden1)
        cur_x, hidden2 = self.rnn2(cur_x, hidden2)
        cur_x, hidden3 = self.rnn3(cur_x, hidden3)
        cur_x = f.sigmoid(self.fc2(cur_x))

        cur_x = cur_x.squeeze(1)

        return cur_x
