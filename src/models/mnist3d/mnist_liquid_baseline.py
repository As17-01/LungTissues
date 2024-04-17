import torch
import torch.nn.functional as f
from ncps.torch import LTC

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST3dLiquidBaseline(BaseModel):
    def __init__(self):
        super(MNIST3dLiquidBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(256, 128)
        self.rnn2 = LTC(128, 64)
        self.rnn3 = LTC(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :, :, :])
            output_t = output_t.unsqueeze(1)
            output = torch.mean((output, output_t ), 1)
        return output



        # hidden1 = torch.zeros(x.shape[0], 128, device=get_default_device())
        # hidden2 = torch.zeros(x.shape[0], 64, device=get_default_device())
        # hidden3 = torch.zeros(x.shape[0], 32, device=get_default_device())

        # cur_x = x.view(x.shape[0], -1)
        # cur_x = f.relu(self.fc1(cur_x))

        # cur_x = cur_x.unsqueeze(1)

        # cur_x, hidden1 = self.rnn1(cur_x, hidden1)
        # cur_x, hidden2 = self.rnn2(cur_x, hidden2)
        # cur_x, hidden3 = self.rnn3(cur_x, hidden3)
        # cur_x = f.sigmoid(self.fc2(cur_x))

        # cur_x = cur_x.squeeze(1)

        # return cur_x
