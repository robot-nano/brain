import csv
import logging
import os
from typing import Optional
import torch

logger = logging.getLogger(__name__)


def length_to_mask(
    length: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
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


def split_word(sequences, space="_"):
    results = []
    for seq in sequences:
        chars = list(space.join(seq))
        results.append(chars)
    return results


def extract_concepts_values(sequences, keep_values, tag_in, tag_out, space):
    """keep the semantic concepts and values for evaluation.

    Arguments
    ---------
    sequences: list
        Each item contains a list, and this list contains a character sequence.
    keep_values: bool
        If True, keep the values. If not don't.
    tag_in: char
        Indicates the start of the concept.
    tag_out: char
        Indicates the end of the concept.
    space: string
        The token represents space. Default: _

    Returns
    -------
    The list contains concept and value sequences for each sentence.

    Example
    -------
    >>> sequences = [['<reponse>','_','n','o','_','>','_','<localisation-ville>','_','L','e','_','M','a','n','s','_','>'], ['<reponse>','_','s','i','_','>'],['v','a','_','b','e','n','e']]
    >>> results = extract_concepts_values(sequences, True, '<', '>', '_')
    >>> results
    [['<reponse> no', '<localisation-ville> Le Mans'], ['<reponse> si'], ['']]
    """
    results = []
    for sequence in sequences:
        # ['<reponse>_no_>_<localisation-ville>_Le_Mans_>']
        sequence = "".join(sequence)
        # ['<reponse>','no','>','<localisation-ville>','Le','Mans,'>']
        sequence = sequence.split(space)
        processed_sequence = []
        value = (
            []
        )  # If previous sequence value never used because never had a tag_out
        kept = ""  # If previous sequence kept never used because never had a tag_out
        concept_open = False
        for word in sequence:
            if re.match(tag_in, word):
                # If not close tag but new tag open
                if concept_open and keep_values:
                    if len(value) != 0:
                        kept += " " + " ".join(value)
                    concept_open = False
                    processed_sequence.append(kept)
                kept = word  # 1st loop: '<reponse>'
                value = []  # Concept's value
                concept_open = True  # Trying to catch the concept's value
                # If we want the CER
                if not keep_values:
                    processed_sequence.append(kept)  # Add the kept concept
            # If we have a tag_out, had a concept, and want the values for CVER
            elif re.match(tag_out, word) and concept_open and keep_values:
                # If we have a value
                if len(value) != 0:
                    kept += " " + " ".join(
                        value
                    )  # 1st loop: '<response>' + ' ' + 'no'
                concept_open = False  # Wait for a new tag_in to pursue
                processed_sequence.append(kept)  # Add the kept concept + value
            elif concept_open:
                value.append(word)  # 1st loop: 'no'
        # If not close tag but end sequence
        if concept_open and keep_values:
            if len(value) != 0:
                kept += " " + " ".join(value)
            concept_open = False
            processed_sequence.append(kept)
        if len(processed_sequence) == 0:
            processed_sequence.append("")
        results.append(processed_sequence)
    return results
