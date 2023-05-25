import sys
import os
import yaml
import tqdm
import logging
import logging.config
import math
import torch
from brain.utils.data_utils import recursive_update
from brain.utils.superpowers import run_shell

ORDERS_ABBREV = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

# Short scale
# Negative powers of ten in lowercase
ORDERS_WORDS = {
    -24: "septillionths",
    -21: "sextillionths",
    -18: "quintillionths",
    -15: "quadrillionths",
    -12: "trillionths",
    -9: "billionths",
    -6: "millionths",
    -3: "thousandths",
    0: "",
    3: "Thousand",
    6: "Million",
    9: "Billion",
    12: "Trillion",
    15: "Quadrillion",
    18: "Quintillion",
    21: "Sextillion",
    24: "Septillion",
}


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration.

    Arguments
    ---------
    config_path : str
        The path to a logging config file.
    default_level : int
        The level to use if the config file is not found.
    overrides : dict
        A dictionary of the same structure as the config dict
        with any updated values that need to be applied.
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def format_order_of_magnitude(number, abbreviate=True):
    style = ORDERS_ABBREV if abbreviate else ORDERS_WORDS
    precision = "{num:3.1f}"
    order = 3 * math.floor(math.log(math.fabs(number), 1000))
    # Fallback for very large numbers:
    while order not in style and order != 0:
        order = order - math.copysign(3, order)  # Bring 3 units towards 0
    order_token = style[order]
    if order != 0:
        formatted_number = precision.format(num=number / 10 ** order)
    else:
        if isinstance(number, int):
            formatted_number = str(number)
        else:
            formatted_number = precision.format(num=number)
    if abbreviate or not order_token:
        return formatted_number + order_token
    else:
        return formatted_number + " " + order_token


def get_environment_description():
    """Returns a string describing the current Python / SpeechBrain environment.

    Useful for making experiments as replicable as possible.

    Returns
    -------
    str
        The string is formatted ready to be written to a file.

    Example
    -------
    >>> get_environment_description().splitlines()[0]
    'SpeechBrain system description'
    """
    python_version_str = "Python version:\n" + sys.version + "\n"
    try:
        freezed, _, _ = run_shell("pip freeze")
        python_packages_str = "Installed Python packages:\n"
        python_packages_str += freezed.decode(errors="replace")
    except OSError:
        python_packages_str = "Could not list python packages with pip freeze"
    try:
        git_hash, _, _ = run_shell("git rev-parse --short HEAD")
        git_str = "Git revision:\n" + git_hash.decode(errors="replace")
    except OSError:
        git_str = "Could not get git revision"
    if torch.cuda.is_available():
        if torch.version.cuda is None:
            cuda_str = "ROCm version:\n" + torch.version.hip
        else:
            cuda_str = "CUDA version:\n" + torch.version.cuda
    else:
        cuda_str = "CUDA not available"
    result = "SpeechBrain system description\n"
    result += "==============================\n"
    result += python_version_str
    result += "==============================\n"
    result += python_packages_str
    result += "==============================\n"
    result += git_str
    result += "==============================\n"
    result += cuda_str
    return result
