import torch


class ModuleList(torch.nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        """Applies the computation pipeline."""
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def append(self, module):
        """Appends module to the layers list."""
        self.layers.append(module)

    def extend(self, modules):
        """Appends module to the layers list."""
        self.layers.extend(modules)

    def insert(self, index, module):
        self.layers.insert(index, module)
