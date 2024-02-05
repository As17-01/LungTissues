import torch


class LSTMBaseline(torch.nn.Module):
    def __init__(self):
        super(LSTMBaseline, self).__init__()
        self.lstm = torch.nn.LSTM(3 * 256 * 256, 120, 2, batch_first=True)
        self.fc1 = torch.nn.Linear(120, 2)

    def forward(self, x):
        num_batches = x.shape[0]
        num_slices = x.shape[1]

        hidden = (
            torch.autograd.Variable(torch.zeros(2, num_batches, 120)),
            torch.autograd.Variable(torch.zeros(2, num_batches, 120)),
        )

        x = x.view(num_batches, num_slices, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x[:, -1, :])

        return x
