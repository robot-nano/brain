from torch.utils.data import Dataset
from brain.dataio.dataio import load_data_csv
from brain.dataio.data_pipeline import DataPipeline
import logging

logger = logging.getLogger(__name__)


class DynamicItemDataset(Dataset):
    def __init__(
        self, data, dynamic_items=None, output_keys=None,
    ):
        if dynamic_items is None:
            dynamic_items = []
        if output_keys is None:
            output_keys = []

        self.data = data
        self.data_ids = list(self.data.keys())
        static_keys = list(self.data[self.data_ids[0]].keys())
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
        self.pipeline = DataPipeline(static_keys, dynamic_items)
        self.set_output_keys(output_keys)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        return self.pipeline.compute_outputs({"id": data_id, **data_point})

    def add_dynamic_item(self, func, takes=None, provides=None):
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_keys(self, keys):
        self.pipeline.set_output_keys(keys)

    @classmethod
    def from_csv(
        cls, csv_path, dynamic_items=None, output_keys=None
    ):
        if dynamic_items is None:
            dynamic_items = []
        if output_keys is None:
            output_keys = []
        data = load_data_csv(csv_path)
        return cls(data, dynamic_items, output_keys)


def add_dynamic_item(datasets, func, takes=None, provides=None):
    """Helper for adding the same item to multiple datasets."""
    for dataset in datasets:
        dataset.add_dynamic_item(func, takes, provides)


def set_output_keys(datasets, output_keys):
    """Helper for setting the same item to multiple datasets."""
    for dataset in datasets:
        dataset.set_output_keys(output_keys)
