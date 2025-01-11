import unittest
from copy import deepcopy
from dataclasses import dataclass
import pytest
from typing import Optional
from unittest.mock import patch

import torch

from fairseq.models.ema import EMA


class DummyModule(torch.nn.Module):
    def __init__(self) -> None:
        """LightningModule for testing purposes

        Args:
            epoch_min_loss_override (int, optional): Pass in an epoch that will be set to the minimum
                validation loss for testing purposes (zero based). If None this is ignored. Defaults to None.
        """
        super().__init__()
        self.layer = torch.nn.Linear(in_features=32, out_features=2)
        self.another_layer = torch.nn.Linear(in_features=2, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.another_layer(x)