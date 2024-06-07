import torch
import torch.nn.functional as f
from ncps.torch import LTC
from ncps.wirings import FullyConnected

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST3dCNNExp2N1(BaseModel):
    def __init__(self):
        super(MNIST3dCNNExp2N1, self).__init__()
        self.name = "MNIST3dCNNExp2N1"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = f.relu(self.fc3(cur_x))
            cur_x = f.relu(self.fc4(cur_x))
            cur_x = self.fc5(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)
        return f.sigmoid(torch.mean(output, 1))

class MNIST3dLiqExp2N2(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp2N2, self).__init__()
        self.name = "MNIST3dLiqExp2N2"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, FullyConnected(21, 1))

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



class MNIST3dCNNExp2N3(BaseModel):
    def __init__(self):
        super(MNIST3dCNNExp2N3, self).__init__()
        self.name = "MNIST3dCNNExp2N3"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.fc3(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dCNNExp2N4(BaseModel):
    def __init__(self):
        super(MNIST3dCNNExp2N4, self).__init__()
        self.name = "MNIST3dCNNExp2N4"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 30)
        self.fc2 = torch.nn.Linear(30, 21)
        self.fc3 = torch.nn.Linear(21, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.fc3(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dLiqExp2N5(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp2N5, self).__init__()
        self.name = "MNIST3dLiqExp2N5"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, FullyConnected(21, 14))
        self.rnn2 = LTC(14, FullyConnected(7, 1))

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
    
class MNIST3dLiqExp2N6(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp2N6, self).__init__()
        self.name = "MNIST3dLiqExp2N6"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, FullyConnected(30, 1))

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

