import torch
import torch.nn.functional as f


class LSTMBaseline(torch.nn.Module):
    def __init__(self):
        super(LSTMBaseline, self).__init__()
        self.lstm = torch.nn.LSTM(3 * 299 * 299, 120, 2, batch_first=True)
        self.fc1 = torch.nn.Linear(120, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        slide_size = x.shape[1]

        hidden = (
            torch.autograd.Variable(torch.zeros(2, batch_size, 120)),
            torch.autograd.Variable(torch.zeros(2, batch_size, 120)),
        )

        x = x.view(batch_size, slide_size, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x[:, -1, :])
        x = f.sigmoid(x)

        return x
