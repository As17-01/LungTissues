import torch
import torch.nn.functional as f
from ncps.torch import LTC


class LiquidBaseline(torch.nn.Module):
    # See https://github.com/mlech26l/ncps
    def __init__(self):
        super(LiquidBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 256 * 256, 400)
        self.rnn = LTC(400, 28)
        self.fc2 = torch.nn.Linear(28, 2)

    def forward(self, x):
        num_batches = x.shape[0]
        num_slices = x.shape[1]

        hidden = torch.zeros(num_batches, 28)

        x = x.view(num_batches, num_slices, -1)
        x = self.fc1(x)
        x, hidden = self.rnn(x, hidden)
        x = self.fc2(x[:,-1,:])

        return x
