import torch
import torch.nn.functional as f
from ncps.torch import LTC


class LiquidBaseline(torch.nn.Module):
    # See https://github.com/mlech26l/ncps
    def __init__(self):
        super(LiquidBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 299 * 299, 512)
        self.rnn = LTC(512, 28)
        self.fc2 = torch.nn.Linear(28, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        slide_size = x.shape[1]

        hidden = torch.zeros(batch_size, 28)

        x = x.view(batch_size, slide_size, -1)
        x = f.relu(self.fc1(x))
        x, hidden = self.rnn(x, hidden)
        x = f.relu(x)
        x = self.fc2(x[:, -1, :])
        x = f.sigmoid(x)

        return x
