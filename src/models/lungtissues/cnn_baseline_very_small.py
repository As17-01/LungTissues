import torch
import torch.nn.functional as f

from src.models.base import BaseModel
from src.utils import get_default_device


class CNNBaselineVerySmall(BaseModel):
    def __init__(self):
        super(CNNBaselineVerySmall, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv7 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv8 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv10 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 256)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(256, 128)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.fc3 = torch.nn.Linear(128, 128)
        self.dropout3 = torch.nn.Dropout(0.25)
        self.fc4 = torch.nn.Linear(128, 64)
        self.dropout4 = torch.nn.Dropout(0.25)
        self.fc5 = torch.nn.Linear(64, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([], device=get_default_device())
        for i in range(time_steps):
            cur_x = f.relu(self.conv1(x[:, i, :, :, :]))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))

            cur_x = f.relu(self.conv3(cur_x))
            cur_x = f.max_pool2d(f.relu(self.conv4(cur_x)), (2, 2))

            cur_x = f.relu(self.conv5(cur_x))
            cur_x = f.max_pool2d(f.relu(self.conv6(cur_x)), (2, 2))

            cur_x = f.relu(self.conv7(cur_x))
            cur_x = f.max_pool2d(f.relu(self.conv8(cur_x)), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv9(cur_x)), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv10(cur_x)), (2, 2))

            cur_x = cur_x.view(batch_size, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = self.dropout1(cur_x)
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.dropout2(cur_x)
            cur_x = f.relu(self.fc3(cur_x))
            cur_x = self.dropout3(cur_x)
            cur_x = f.relu(self.fc4(cur_x))
            cur_x = self.dropout4(cur_x)
            cur_x = self.fc5(cur_x)

            cur_x = cur_x.unsqueeze(1)
            output = torch.cat((output, cur_x), 1)

        output = f.sigmoid(torch.mean(output, 1))
        return torch.squeeze(output, 1)
