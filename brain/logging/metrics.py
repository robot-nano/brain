import contextlib
import uuid
from collections import defaultdict
from typing import Callable, List, Optional

from .meters import *

def aggregate(name: Optional[str] = None, new_root: bool = False):
    """Context manager to aggregate metrics under a given name.

    Aggregations can be nested. If *new_root* is ``False``, then logged
    metrics will be recorded along the entire stack of nested
    aggregators, including a global "default" aggregator. If *new_root*
    is ``True``, then this aggregator will be the root of a new
    aggregation stack, thus bypassing any parent aggregators.

    Note that aggregation contexts are uniquely identified by their
    *name* (e.g., train, valid). Creating a context with an existing
    name will reuse the corresponding :class:`MetersDict` instance.
    If no name is given, then a temporary aggregator will be created.

    Usage::

        with metrics.aggregate("train"):
            for step, batch in enumerate(epoch):
                with metrics.aggregate("train_inner") as agg:
                    metrics.log_scalar("loss", get_loss(batch))
                    if step % log_interval == 0:
                        print(agg.get_smoothed_value("loss"))
                        agg.reset()
        print(metrics.get_smoothed_values("train")["loss"])

    Args:
        name (str): name of the aggregation. Defaults to a
            random/temporary name if not given explicitly.
        new_root (bool): make this aggregation the root of a new
            aggregation stack.
    """
    if name is None:
        # generate a temporary name
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = MetersDict()
    else:
        assert name != "default"
        agg = _aggregators.setdefault(name, MetersDict())

    if new_root:
        backup_aggregators = _active_aggregators.copy()
        _active_aggregators.clear()
        backup_aggregators_cnt = _active_aggregators_cnt.copy()
        _active_aggregators_cnt.clear()

    _active_aggregators[name] = agg
    _active_aggregators_cnt[name] += 1

    yield agg

    _active_aggregators_cnt[name] -= 1
    if _active_aggregators_cnt[name] == 0 and name in _active_aggregators:
        del _active_aggregators[name]

    if new_root:
        _active_aggregators.clear()
        _active_aggregators.update(backup_aggregators)
        _active_aggregators_cnt.clear()
        _active_aggregators_cnt.update(backup_aggregators_cnt)
