import torch
import torch.nn.functional as f

from src.models.base import BaseModel


class MNIST2dLinearBaseline(BaseModel):
    def __init__(self):
        super(MNIST2dLinearBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 1)

    def forward(self, x):
        cur_x = x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))
        cur_x = f.relu(self.fc2(cur_x))
        cur_x = f.relu(self.fc3(cur_x))
        cur_x = f.relu(self.fc4(cur_x))
        cur_x = f.sigmoid(self.fc5(cur_x))
        return cur_x
