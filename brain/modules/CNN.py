from collections import OrderedDict
import torch
from torch import nn
from brain.modules.normalization import LayerNorm


class Conv2d(nn.Module):
    def __init__(
        self,
        num_layers,
        channels,
        kernel_sizes,
        strides=1,
        padding="same",
        dilations=1,
        bias=True,
        activation=torch.nn.LeakyReLU,
        norm=None,
        dropout=0.1,
    ):
        super().__init__()
        assert padding == "same" or padding == "valid"
        assert len(channels) == num_layers + 1

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes for _ in range(num_layers)]
        if isinstance(strides, int):
            strides = [strides for _ in range(num_layers)]
        if isinstance(dilations, int):
            dilations = [dilations for _ in range(num_layers)]

        layers = []
        for i in range(num_layers):
            if padding == "same":
                pad_size = kernel_sizes[i] // 2
            else:
                pad_size = 0
            layers.append(
                (
                    f"conv2d_{i}",
                    torch.nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=pad_size,
                        dilation=dilations[i],
                        bias=bias,
                    )
                )
            )
            if norm is not None:
                layers.append(
                    [
                        f"norm_{i}",
                        LayerNorm()
                    ]
                )
            layers.append(
                (
                    f"act_{i}",
                    activation()
                )
            )
            layers.append(
                (
                    f"dropout_{i}",
                    torch.nn.Dropout(dropout)
                )
            )

        self.conv = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.conv(x)
        return out


def infer_conv_output_dim(in_channels, input_dim, out_channels):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
    x = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = x.transpose(1, 2)
    mb, seq = x.size()[:2]
    return x.contiguous().view(mb, seq, -1).size(-1)