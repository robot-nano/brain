import pathlib

import torch
import inspect
import logging
from packaging import version

logger = logging.getLogger(__name__)

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"  # Important that this is not .ckpt
PARAMFILE_EXT = ".ckpt"  # ...because these files will be


def torch_recovery(obj, path, end_of_epoch, device=None):
    del end_of_epoch
    try:
        obj.load_state_dict(torch.load(path, map_location=device), strict=True)
    except TypeError:
        obj.load_state_dict(torch.load(path, map_location=device))


def torch_save(obj, path):
    state_dict = obj.state_dict()
    if not state_dict:
        logger.warning(f"Saving an empty state_dict for {obj} in {path}.")
    torch.save(state_dict, path)


def torch_parameter_transfer(obj, path, device):
    incompatible_keys = obj.load_state_dict(
        torch.load(path, map_location=device), strict=False
    )
    for missing_key in incompatible_keys.missing_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the transferred parameters did not have "
            + f"parameters for the key: {missing_key}"
        )
    for unexpected_key in incompatible_keys.unexpected_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the object could not use the parameters loaded "
            + f"with the key: {unexpected_key}"
        )


if version.parse(torch.__version__) < version.parse("2.0.0"):
    DEFAULT_LOAD_HOOKS = {
        torch.nn.Module: torch_recovery,
        torch.optim.Optimizer: torch_recovery,
        torch.optim.lr_scheduler._LRScheduler: torch_recovery,
        torch.optim.lr_scheduler.ReduceLROnPlateau: torch_recovery,
        torch.cuda.amp.grad_scaler.GradScaler: torch_recovery,
    }
    DEFAULT_SAVE_HOOKS = {
        torch.nn.Module: torch_save,
        torch.optim.Optimizer: torch_save,
        torch.optim.lr_scheduler._LRScheduler: torch_save,
        torch.optim.lr_scheduler.ReduceLROnPlateau: torch_save,
        torch.cuda.amp.grad_scaler.GradScaler: torch_save,
    }
else:
    DEFAULT_LOAD_HOOKS = {
        torch.nn.Module: torch_recovery,
        torch.optim.Optimizer: torch_recovery,
        torch.optim.lr_scheduler.LRScheduler: torch_recovery,
        torch.optim.lr_scheduler.ReduceLROnPlateau: torch_recovery,
        torch.cuda.amp.grad_scaler.GradScaler: torch_recovery,
    }
    DEFAULT_SAVE_HOOKS = {
        torch.nn.Module: torch_save,
        torch.optim.Optimizer: torch_save,
        torch.optim.lr_scheduler.LRScheduler: torch_save,
        torch.optim.lr_scheduler.ReduceLROnPlateau: torch_save,
        torch.cuda.amp.grad_scaler.GradScaler: torch_save,
    }

DEFAULT_TRANSFER_HOOKS = {
    torch.nn.Module: torch_parameter_transfer,
}

# Add a transfer hook for sentencepiece if it is installed:
try:
    import sentencepiece as spm


    def _load_spm(obj, path, device=None):
        obj.load(str(path))  # SentencePieceProcessor needs a string.


    DEFAULT_TRANSFER_HOOKS[spm.SentencePieceProcessor] = _load_spm
    del spm
except ImportError:
    # SentencePiece not loaded, fine!
    pass


def mark_as_saver(method):
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Checkpoint saver must match signature (instance, path)"
        raise TypeError(MSG)
    method._brain_saver = True
    return method


def mark_as_loader(method):
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"), True, None)
    except TypeError:
        MSG = "Checkpoint loader must have signature (self, path, end_of_epoch, device)"
        raise TypeError(MSG)
    method._brain_loader = True
    return method


def mark_as_transfer(method):
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"), device=None)
    except TypeError:
        MSG = "Transfer hook must have signature (self, path, device)"
        raise TypeError(MSG)
    method._brain_transfer = True
    return method


def register_checkpoint_hooks(cls):
    global DEFAULT_LOAD_HOOKS
    global DEFAULT_SAVE_HOOKS
    global DEFAULT_TRANSFER_HOOKS
    for name, method in cls.__dict__.items():
        if hasattr(method, "_brain_saver"):
            DEFAULT_SAVE_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint save hook for {name}")
        if hasattr(method, "_brain_loader"):
            DEFAULT_LOAD_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint load hook for {name}")
        if hasattr(method, "_brain_transfer"):
            DEFAULT_TRANSFER_HOOKS[cls] = method
            logger.debug(f"Registered parameter transfer hook for {name}")
    return cls


def get_default_hook(obj, default_hooks):
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    # If we got there, no hook found
    return None
