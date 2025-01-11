import unittest
from copy import deepcopy
from dataclasses import dataclass
import pytest
from typing import Optional
from unittest.mock import patch

import torch


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


class EMAConfig(object):
    ema_decay: float = 0.99
    ema_start_update: int = 0
    ema_fp32: bool = False
    ema_seed_model: Optional[str] = None
    ema_update_freq: int = 1


class TestEMA(unittest.TestCase):
    def assertTorchAllClose(self, x, y, atol=1e-8, rtol=1e-5, msg=None):
        diff = x.float() - y.float()
        diff_norm = torch.norm(diff)
        other_norm = torch.norm(y.float())

        if msg is None:
            msg = "|input - other| > {} + {} * |other|".format(atol, rtol)

        self.assertLessEqual(
            diff_norm,
            atol + rtol * other_norm,
            msg=msg,
        )

    def test_ema(self):
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig()
        ema = EMA(model, config)

        # set decay
        ema._set_decay(config.ema_decay)
        self.assertEqual(ema.get_decay(), config.ema_decay)

        # get model
        self.assertEqual(ema.get_model(), ema.model)

        # Since fp32 params is not used, it should be of size 0
        self.assertEqual(len(ema.fp32_params), 0)

        # EMA step
        x = torch.randn(32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        ema.step(model)

        ema_state_dict = ema.get_model().state_dict()

        for key, param in model.state_dict().items():
            prev_param = state[key]
            ema_param = ema_state_dict[key]

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue
            self.assertTorchAllClose(
                ema_param,
                config.ema_decay * prev_param + (1 - config.ema_decay) * param,
            )

        # Since fp32 params is not used, it should be of size 0
        self.assertEqual(len(ema.fp32_params), 0)

        # Load EMA into model
        model2 = DummyModule()
        ema.reverse(model2)

        for key, param in model2.state_dict().items():
            ema_param = ema_state_dict[key]
            self.assertTrue(torch.allclose(ema_param, param))

        # Check that step_internal is called once
        with patch.object(ema, "_step_internal", return_value=None) as mock_method:
            ema.step(model)
            mock_method.assert_called_once_with(model, None)

    def _test_ema_start_update(self, updates):
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig(ema_start_update=1)
        ema = EMA(model, config)

        # EMA step
        x = torch.randn(32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        ema.step(model, updates=updates)
        ema_state_dict = ema.get_model().state_dict()

        self.assertEqual(ema.get_decay(), 0 if updates == 0 else config.ema_decay)

        for key, param in model.state_dict().items():
            ema_param = ema_state_dict[key]
            prev_param = state[key]

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue
            if updates == 0:
                self.assertTorchAllClose(
                    ema_param,
                    param,
                )
            else:
                self.assertTorchAllClose(
                    ema_param,
                    config.ema_decay * prev_param + (1 - config.ema_decay) * param,
                )

        # Check that step_internal is called once
        with patch.object(ema, "_step_internal", return_value=None) as mock_method:
            ema.step(model, updates=updates)
            mock_method.assert_called_once_with(model, updates)