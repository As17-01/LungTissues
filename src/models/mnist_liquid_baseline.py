import torch
import torch.nn.functional as f
from ncps.torch import LTC

from src.models.base import BaseModel
from src.utils import get_default_device


class MNISTLiquidBaseline(BaseModel):
    def __init__(self):
        super(MNISTLiquidBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 40)
        self.rnn = LTC(40, 28)
        self.fc2 = torch.nn.Linear(28, 1)

    def forward(self, x):
        hidden = torch.zeros(x.shape[0], 28, device=get_default_device())

        cur_x = x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))

        cur_x = cur_x.unsqueeze(1)

        cur_x, hidden = self.rnn(cur_x, hidden)
        cur_x = f.sigmoid(self.fc2(cur_x))

        cur_x = cur_x.squeeze(1)

        return cur_x
