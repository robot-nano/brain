from operator import itemgetter
import torch
from torch.utils.data import (
    RandomSampler,
    DistributedSampler,
    Sampler,
)


class ReproducibleRandomSampler(RandomSampler):
    def __init__(self, data_source, seed=563375142, epoch=0, **kwargs):
        if "generator" in kwargs:
            MSG = {
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            }
            raise ValueError(MSG)
        super().__init__(data_source, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        self.generator = torch.Generator()
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(dataset=sampler, *args, **kwargs)
        self.sampler = sampler
    
    def __iter__(self):
        # It is easiest to use a random access interface to the wrapped
        # sampler's indices, so we just fetch all indices from the wrappered
        # sampler
        sampler_indices = list(self.sampler.__iter__())
        indices_of_indices = super().__iter__()
        # Itemgetter fetches the wrapped sampler indices from the positions
        # pointed to by DistributedSampler
        return iter(itemgetter(*indices_of_indices)(sampler_indices))

    def set_epoch(self, epoch: int):
        """Pass set_epoch() through to DistributedSampler and the wrapper one"""
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
