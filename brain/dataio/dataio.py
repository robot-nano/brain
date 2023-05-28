import csv
import logging
import os

import torch

logger = logging.getLogger(__name__)


def length_to_mask(
    length: torch.tensor,
    max_len=None,
    dtype=None,
    device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(length.shape[0], max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def load_data_csv(csv_path):
    with open(csv_path, "r") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            try:
                data_id = row["ID"]
                del row["ID"]
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")

            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def merge_char(sequences, space="_"):
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return results


def merge_csvs(data_folder, csv_list, merged_csv):
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Complete in previous run.")
    with open(os.path.join(data_folder, csv_list[0])) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_list:
        with open(os.path.join(data_folder, csv_file)) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_list[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")
