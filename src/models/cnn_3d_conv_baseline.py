import torch
import torch.nn.functional as f


class CNN3DConvBaseline(torch.nn.Module):
    def __init__(self):
        super(CNN3DConvBaseline, self).__init__()
        self.conv1 = torch.nn.Conv3d(64, 6, (1, 5, 5))
        self.conv2 = torch.nn.Conv3d(6, 16, (1, 3, 3))
        self.fc1 = torch.nn.Linear(16 * 3 * 62 * 62, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        x = f.max_pool3d(f.relu(self.conv1(x)), (1, 2, 2))
        x = f.max_pool3d(f.relu(self.conv2(x)), (1, 2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
