import torch
import torch.nn.functional as f
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST3dCNNExp3N1(BaseModel):
    def __init__(self):
        super(MNIST3dCNNExp3N1, self).__init__()
        self.name = "MNIST3dCNNExp3N1"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)
            cur_x = self.dropout1(cur_x)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = self.dropout2(cur_x)
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.dropout3(cur_x)
            cur_x = self.fc3(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)
        return f.sigmoid(torch.mean(output, 1))

class MNIST3dLiqExp3N2(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp3N2, self).__init__()
        self.name = "MNIST3dLiqExp3N2"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 256)
        self.rnn1 = LTC(256, AutoNCP(21, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = self.dropout1(cur_x)
            cur_x = f.relu(self.fc1(cur_x))

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = self.rnn1(output)[0]
        return f.sigmoid(torch.mean(output, 1))



class MNIST3dLiqExp3N3(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp3N3, self).__init__()
        self.name = "MNIST3dLiqExp3N3"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.bn1 = torch.nn.BatchNorm1d(16 * 5 * 5)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(21, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = self.bn1(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = self.dropout1(output)
        output = self.rnn1(output)[0]
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dLiqExp3N4(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp3N4, self).__init__()
        self.name = "MNIST3dLiqExp3N4"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(21, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = self.dropout1(output)
        output = self.rnn1(output)[0]
        return f.sigmoid(torch.mean(output, 1))


class MNIST3dLiqExp3N5(BaseModel):
    def __init__(self):
        super(MNIST3dLiqExp3N5, self).__init__()
        self.name = "MNIST3dLiqExp3N5"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn1 = LTC(16 * 5 * 5, AutoNCP(21, 14))
        self.fc1 = torch.nn.Linear(14, 1)

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
        output = torch.mean(output, 1)
        output = self.fc1(output)

        return f.sigmoid(output)
    
class MNIST3dCNNExp3N6(BaseModel):
    def __init__(self):
        super(MNIST3dCNNExp3N6, self).__init__()
        self.name = "MNIST3dCNNExp3N6"
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.max_pool2d(f.relu(self.conv1(x[:, i, :, :, :])), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
            cur_x = cur_x.view(batch_size, -1)
            cur_x = self.dropout1(cur_x)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = self.dropout2(cur_x)
            cur_x = self.fc2(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = f.relu(torch.mean(output, 1))
        output = self.dropout3(output)
        output = self.fc3(output)
        return f.sigmoid(output)
