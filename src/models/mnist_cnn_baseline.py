import torch
import torch.nn.functional as f

from src.models.base import BaseModel


class MNISTCNNBaseline(BaseModel):
    def __init__(self):
        super(MNISTCNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        cur_x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
        cur_x = cur_x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))
        cur_x = f.relu(self.fc2(cur_x))
        cur_x = f.sigmoid(self.fc3(cur_x))
        return cur_x
