from torch.utils.data import DataLoader, IterableDataset
from brain.dataio.dataset import DynamicItemDataset
from brain.dataio.batch import PaddedBatch
from brain.dataio.sampler import ReproducibleRandomSampler
import logging
from brain.utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

try:
    import webdataset as wds
    from importlib_metadata import version

    WDS_AVAILABLE = True

    # Use appropriate class based on webdataset version
    if version("webdataset")[0:4] == "0.1.":
        WDS_CLASS = wds.dataset.Composable
    else:
        WDS_CLASS = wds.DataPipeline
except ImportError:
    WDS_AVAILABLE = False

logger = logging.getLogger(__name__)


def make_dataloader(dataset, looped_nominal_epoch=None, **loader_kwargs):
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch

    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]
    # with WDS it is recommended to do batching in the dataset itself,
    # which requires batch_size = None in the Dataloader
    if (
        WDS_AVAILABLE
        and isinstance(dataset, WDS_CLASS)
        and "batch_size" not in loader_kwargs
    ):
        loader_kwargs["batch_size"] = None
    # Create the loader
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        dataloader = SavableDataLoader(dataset, **loader_kwargs)
    return dataloader


@register_checkpoint_hooks
class SavableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logger.warning(
                "SavableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._brain_recovery_skip_to = None
        self._brain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._brain_iterator = iterator
        return iterator

    @mark_as_saver
    def _brain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._brain_iterator is None:
            to_save = None
        else:
            to_save = self._brain_iterator._num_yielded
        with open(path, "w") as fo:
            fo.write(str(to_save))

    @mark_as_loader
    def _brain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._brain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._brain_recovery_skip_to = int(saved)
