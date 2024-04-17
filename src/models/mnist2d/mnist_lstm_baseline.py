import torch
import torch.nn.functional as f

from src.models.base import BaseModel
from src.utils import get_default_device


class MNIST2dLSTMBaseline(BaseModel):
    def __init__(self):
        super(MNIST2dLSTMBaseline, self).__init__()
        self.lstm = torch.nn.LSTM(28 * 28, 40, 2, batch_first=True)
        self.fc1 = torch.nn.Linear(40, 1)

    def forward(self, x):
        # It resets hidden state. It is usless until we deal with series data.
        hidden = (
            torch.autograd.Variable(torch.zeros(2, 40, device=get_default_device())),
            torch.autograd.Variable(torch.zeros(2, 40, device=get_default_device())),
        )
        cur_x = x.view(x.shape[0], -1)

        cur_x, hidden = self.lstm(cur_x, hidden)
        cur_x = f.sigmoid(self.fc1(cur_x))

        return cur_x
