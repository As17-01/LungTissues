import torch

class LiquidBaseline(torch.nn.Module):
    # See https://github.com/mlech26l/ncps
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def num_flat_features(self, x):
        raise NotImplementedError
