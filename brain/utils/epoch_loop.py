from .checkpoints import register_checkpoint_hooks
from .checkpoints import mark_as_saver
from .checkpoints import mark_as_loader
import logging

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class EpochCounter:
    def __init__(self, limit):
        self.current = 0
        self.limit = int(limit)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            logger.info(f"Going into epoch {self.current}")
            return self.current
        raise StopIteration

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @mark_as_loader
    def _recover(self, path, end_of_epoch=True, device=None):
        # NOTE: end_of_epoch = True by default to that when
        # loaded in parameter transfer, this starts a new epoch.
        # However, parameter transfer to EpochCounter should
        # probably never be used really.
        del device  # Not used.
        with open(path) as fi:
            saved_value = int(fi.read())
            if end_of_epoch:
                self.current = saved_value
            else:
                self.current = saved_value - 1
