import torch
import torch.nn.functional as f


class CNNBaseline(torch.nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 62 * 62, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        num_batches = x.shape[0]
        num_slices = x.shape[1]

        scores = torch.tensor([[0.0, 0.0]] * num_batches)

        x = x.permute(1, 0, 2, 3, 4)
        for cur_x in x:
            cur_x = f.max_pool2d(f.relu(self.conv1(cur_x)), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), 2)
            cur_x = cur_x.view(num_batches, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.fc3(cur_x)

            scores += torch.div(cur_x, num_slices)
        return scores
