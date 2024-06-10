import torch
import torch.nn.functional as f
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST3dLiqExp5N1(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp5N1, self).__init__()
        self.name = "MNIST3dLiqExp5N1"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(5, 1))

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


class MNIST3dLSTMExp5N2(BaseModel):
    def __init__(self):
        super(MNIST3dLSTMExp5N2, self).__init__()
        self.name = "MNIST3dLSTMExp5N2"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = torch.nn.LSTM(16 * 5 * 5, 40, 1)
        self.fc1 = torch.nn.Linear(40, 1)

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
        output = self.fc1(output)
        return f.sigmoid(torch.mean(output, 1))
    
class MNIST3dLSTMExp5N3(BaseModel):
    def __init__(self):
        super(MNIST3dLSTMExp5N3, self).__init__()
        self.name = "MNIST3dLSTMExp5N3"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = torch.nn.LSTM(16 * 5 * 5, 40, 1)
        self.rnn2 = torch.nn.LSTM(40, 1, 1)

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
        output = self.rnn2(output)[0]
        return f.sigmoid(torch.mean(output, 1))
    

class MNIST3dLSTMExp5N4(BaseModel):
    def __init__(self):
        super(MNIST3dLSTMExp5N4, self).__init__()
        self.name = "MNIST3dLSTMExp5N4"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = torch.nn.LSTM(16 * 5 * 5, 40, 2)
        self.fc1 = torch.nn.Linear(40, 1)

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
        output = self.fc1(output)
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dRNNExp5N5(BaseModel):
    def __init__(self):
        super(MNIST3dRNNExp5N5, self).__init__()
        self.name = "MNIST3dRNNExp5N5"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = torch.nn.RNN(16 * 5 * 5, 40, 1)
        self.fc1 = torch.nn.Linear(40, 1)

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
        output = self.fc1(output)
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dRNNExp5N6(BaseModel):
    def __init__(self):
        super(MNIST3dRNNExp5N6, self).__init__()
        self.name = "MNIST3dRNNExp5N6"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = torch.nn.RNN(16 * 5 * 5, 40, 2)
        self.fc1 = torch.nn.Linear(40, 1)

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
        output = self.fc1(output)
        return f.sigmoid(torch.mean(output, 1))
