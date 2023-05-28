import logging
import pathlib

from brain.pretrained.fetching import fetch
from brain.utils.checkpoints import (
    DEFAULT_LOAD_HOOKS,
    DEFAULT_TRANSFER_HOOKS,
    PARAMFILE_EXT,
    get_default_hook,
)

logger = logging.getLogger(__name__)


class Pretrainer:
    def __init__(
        self,
        collect_in="./model_checkpoints",
        loadables=None,
        paths=None,
        custom_hooks=None,
        conditions=None,
    ):
        self.loadables = {}
        self.collect_in = pathlib.Path(collect_in)
        if loadables is not None:
            self.add_loadables(loadables)
        self.paths = {}
        if paths is not None:
            self.add_paths(paths)
        self.custom_hooks = {}
        if custom_hooks is not None:
            self.add_custom_hooks(custom_hooks)
        self.conditions = {}
        if conditions is not None:
            self.add_conditions(conditions)

    def set_collect_in(self, path):
        """Change the collecting path"""
        self.collect_in = pathlib.Path(path)

    def add_loadables(self, loadables):
        self.loadables.update(loadables)

    def add_paths(self, paths):
        self.paths.update(paths)

    def add_custom_hooks(self, custom_hooks):
        self.custom_hooks.update(custom_hooks)

    def add_conditions(self, conditions):
        self.conditions.update(conditions)

    @staticmethod
    def split_path(path):
        if "/" in path:
            return path.rsplit("/", maxsplit=1)
        else:
            # Interpret as path to file in current directory.
            return "./", path

    def collect_files(self, default_source=None):
        logger.debug(
            f"Collecting files (or symlinks) for pretraining in {self.collect_in}."
        )
        self.collect_in.mkdir(exist_ok=True)
        loadable_paths = {}
        for name in self.loadables:
            if not self.is_loadable(name):
                continue
            save_filename = name + PARAMFILE_EXT
            if name in self.paths:
                source, filename = self.split_path(self.paths[name])
            elif default_source is not None:
                filename = save_filename
                source = default_source
            else:
                raise ValueError(
                    f"Path not specified for '{name}', "
                    "and no default_source given!"
                )
            path = fetch(
                filename=filename,
                source=source,
                savedir=self.collect_in,
                overwrite=False,
                save_filename=save_filename,
                use_auth_token=False,
                revision=None,
            )
            loadable_paths[name] = path
        return loadable_paths

    def is_loadable(self, name):
        if name not in self.conditions:
            return True
        condition = self.conditions[name]
        if callable(condition):
            return condition()
        else:
            return bool(condition)

    def load_collected(self, device=None):
        logger.info(
            f"Loading pretrained files for: {', '.join(self.loadables)}"
        )
        paramfiles = {}
        for name in self.loadables:
            if not self.is_loadable(name):
                continue
            filename = name + PARAMFILE_EXT
            paramfiles[name] = self.collect_in / filename
        self._call_load_hooks(paramfiles, device)

    def _call_load_hooks(self, paramfiles, device=None):
        for name, obj in self.loadables.items():
            if not self.is_loadable(name):
                continue
            loadpath = paramfiles[name]

            # First see if object has custom load hook:
            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath, device=device)
                continue
            # Try the default transfer hook:
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath, device=device)
                continue
            # Otherwise find the default loader for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                # Need to fake end-of-epoch:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch, device)
                continue
            # If we got here, no custom hook or registered default hook exists
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                  or add custom hook for this object."
            return RuntimeError(MSG)
