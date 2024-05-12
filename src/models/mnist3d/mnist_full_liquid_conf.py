import torch
import torch.nn.functional as f
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST3dFullLiquidConf0(BaseModel):
    def __init__(self):
        super(MNIST3dFullLiquidConf0, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(128, 84))
        self.rnn2 = LTC(84, AutoNCP(64, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = f.relu(self.rnn1(output)[0])
        output = self.rnn2(output)[0]
        return f.sigmoid(torch.mean(output, 1))
    

class MNIST3dFullLiquidConf1(BaseModel):
    def __init__(self):
        super(MNIST3dFullLiquidConf1, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(128, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = self.rnn1(output)[0]
        return f.sigmoid(torch.mean(output, 1))

class MNIST3dFullLiquidConf2(BaseModel):
    def __init__(self):
        super(MNIST3dFullLiquidConf2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(400, 120))
        self.rnn2 = LTC(120, AutoNCP(120, 84))
        self.rnn3 = LTC(84, AutoNCP(84, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = f.relu(self.rnn1(output)[0])
        output = f.relu(self.rnn2(output)[0])
        output = self.rnn3(output)[0]
        return f.sigmoid(torch.mean(output, 1))

