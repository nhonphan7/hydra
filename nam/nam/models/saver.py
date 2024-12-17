"""Utility classes for saving model checkpoints."""
import inspect
import os

import torch
import torch.nn as nn


class Checkpointer:
    """A simple `PyTorch` model load/save wrapper."""

    def __init__(
        self,
        log_dir: str = 'output',
        device: str = 'cpu'
    ) -> None:
        """Constructs a simple load/save checkpointer."""
        self._ckpt_dir = os.path.join(log_dir, "ckpts")
        self._device = device
        os.makedirs(self._ckpt_dir, exist_ok=True)

    def save(
        self,
        model,
        epoch: int,
    ) -> str:
        """Saves the model to the `ckpt_dir/epoch/model.pt` file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        torch.save({
            'model_state_dict': model.state_dict(),
            'attributes': vars(model),
            'class': type(model)
            }, 
            ckpt_path
        )
        return ckpt_path

    def load(
        self,
        epoch: int,
    ) -> nn.Module:
        """Loads the model from the `ckpt_dir/epoch/model.pt` file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        ckpt = torch.load(ckpt_path, map_location=self._device)
        constructor = ckpt['class']
        constructor_args = inspect.getfullargspec(constructor).args
        args = {k: v for k, v in ckpt['attributes'].items() if k in constructor_args}
        model = constructor(**args)
        model.load_state_dict(ckpt['model_state_dict'])
        return model
