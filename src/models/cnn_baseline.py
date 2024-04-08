import torch
import torch.nn.functional as f

from src.models.base import BaseModel


class CNNBaseline(BaseModel):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.conv6 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000)
        self.fc2 = torch.nn.Linear(1000, 1)

    def forward(self, x):
        cur_x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv3(cur_x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv4(cur_x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv5(cur_x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv6(cur_x)), (2, 2))

        cur_x = cur_x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))
        cur_x = f.sigmoid(self.fc2(cur_x))
        return cur_x
