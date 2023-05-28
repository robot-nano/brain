import torch


class Swish(torch.nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.beta * x)
