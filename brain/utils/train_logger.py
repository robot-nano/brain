import logging
import ruamel.yaml
import torch
import os

logger = logging.getLogger(__name__)


class TrainLogger:
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        raise NotImplementedError


class FileTrainLogger(TrainLogger):
    def __init__(self, save_file, precision=2):
        self.save_file = save_file
        self.precision = precision

    def _item_to_string(self, key, value, dataset=None):
        """Convert one item to string, handling floats"""
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}e}"
        if dataset is not None:
            key = f"{dataset} {key}"
        return f"{key}: {value}"

    def _stats_to_string(self, stats, dataset=None):
        """Convert all stats to a single string summary"""
        return ", ".join(
            [self._item_to_string(k, v, dataset) for k, v in stats.items()]
        )

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        """See TrainLogger.log_stats()"""
        string_summary = self._stats_to_string(stats_meta)
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is not None:
                string_summary += " - " + self._stats_to_string(stats, dataset)

        with open(self.save_file, "a") as fout:
            print(string_summary, file=fout)
        if verbose:
            logger.info(string_summary)

