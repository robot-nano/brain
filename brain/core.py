import os
import sys
import warnings
import time
import yaml
import tempfile
import logging
import pathlib
import shutil
import inspect
from tqdm import tqdm
from contextlib import contextmanager
from datetime import date
from enum import Enum, auto
from hyperpyyaml import resolve_references
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
import torch
import torch.distributed as dist
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    DistributedSampler,
)
import brain
from brain.utils.distributed import run_on_main
from brain.dataio.dataloader import SavableDataLoader
from brain.dataio.sampler import (
    ReproducibleRandomSampler,
    DistributedSamplerWrapper,
)

logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")
INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
    save_env_desc=True,
):
    """Create the output folder and relevant experimental files.

    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved, and the result is
        written to a file in the experiment directory called "hyperparams.yaml".
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, an environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory.
    """
    try:
        # all writing command must be done with the main_process
        if brain.utils.distributed.if_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Write the parameters file
            if hyperparams_to_save is not None:
                hyperparams_filename = os.path.join(
                    experiment_directory, "hyperparams.yaml"
                )
                with open(hyperparams_to_save) as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w") as w:
                    print("# Generated %s from:" % date.today(), file=w)
                    print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                    print("# yamllint disable", file=w)
                    shutil.copyfileobj(resolved_yaml, w)

            # Copy executing file to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingfile = os.path.realpath(module.__file__)
                shutil.copy(callingfile, experiment_directory)

            # Log exceptions to output automatically
            log_file = os.path.join(experiment_directory, "log.txt")
            logger_overrides = {
                "handlers": {"file_handler": {"filename": log_file}}
            }
            brain.utils.logger.setup_logging(log_config, logger_overrides)
            sys.excepthook = _logging_excepthook

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

            # Save system description:
            if save_env_desc:
                description_str = brain.utils.logger.get_environment_description()
                with open(
                    os.path.join(experiment_directory, "env.log"), "w"
                ) as fo:
                    fo.write(description_str)
    finally:
        # wait for main_process if ddp is used
        brain.utils.distributed.ddp_barrier()


def _logging_excepthook(exc_type, exc_value, exc_traceback):
    """Interrupt exception raising to log the error."""
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))


@brain.utils.checkpoints.register_checkpoint_hooks
class Brain:
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):
        self.opt_class = opt_class
        self.checkpointer = checkpointer
        self.profiler = profiler

        # Arguments passed via the run opts dictionary
        run_opt_defaults = {
            "debug": False,
            "debug_batches": 2,
            "debug_epochs": 2,
            "debug_persistently": False,
            "device": "cpu",
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "find_unused_parameters": False,
            "jit_module_keys": None,
            "auto_mix_prec": False,
            "max_grad_norm": 5.0,
            "nonfinite_patience": 3,
            "ckpt_interval_minutes": 10,
            "grad_accumulation_factor": 1,
            "optimizer_step_limit": None,
            "tqdm_barcolor": {
                "train": "GREEN",
                "valid": "MAGENTA",
                "test": "CYAN",
            },
        }

        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            logger.warning(
                "Detected Python "
                + str(sys.version_info.major)
                + "."
                + str(sys.version_info.minor)
                + ". We suggest using Brain with Python >="
                + str(PYTHON_VERSION_MAJOR)
                + "."
                + str(PYTHON_VERSION_MINOR)
            )

        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Automatic mixed precision init
        if self.auto_mix_prec:
            self.scaler = torch.cuda.amp.GradScaler()
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("scaler", self.scaler)

        # List parameter count for the user
        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        if total_params > 0:
            clsname = self.__class__.__name__
            fmt_num = brain.utils.logger.format_order_of_magnitude(total_params)
            logger.info(f"{fmt_num} trainable parameters in {clsname}")

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not dist.is_initialized():
                if self.rank > 0:
                    sys.exit(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "python -m torch.distributed.launch [args]\n\t"
                        "experiment.py hyperparams.yaml "
                        "--distributed_launch=True --distributed_backend=nccl"
                    )
                else:
                    logger.warning(
                        "To use DDP, please add "
                        "sb.utils.distributed.ddp_init_group() into your exp.py"
                    )
                    logger.info(
                        "Only the main process is alive, "
                        "all other subprocess were killed."
                    )

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

    def compute_forward(self, batch, stage):
        raise NotImplementedError

    def compute_objectives(self, predictions, batch, stage):
        raise NotImplementedError

    def on_stage_start(self, stage, epoch=None):
        pass

    def on_stage_end(self, stage, stage_loss, epoch=None):
        pass

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        """

        Parameters
        ----------
        dataset
        stage
        ckpt_prefix : str, None
        loader_kwargs

        Returns
        -------

        """
        if self.distributed_launch:
            loader_kwargs = self._distributed_specifics(dataset, loader_kwargs)
        dataloader = brain.dataio.dataloader.make_dataloader(
            dataset, **loader_kwargs
        )

        if (
            self.checkpointer is not None
            and ckpt_prefix is not None
            and isinstance(dataloader, SavableDataLoader)
        ):
            ckpt_key = ckpt_prefix + stage.name
            self.checkpointer.add_recoverable(ckpt_key, dataloader)
        return dataloader

    def _distributed_specifics(self, dataset, loader_kwargs):
        sampler = loader_kwargs.get("sampler", None)
        # Shuffling should really only matter for the train stage. Shuffling
        # will also lead to padding in batches if the order was otherwise
        # sorted by length.
        shuffle = loader_kwargs.get("shuffle", False)

        # Possibly make a DistributedSampler or a wrapper for some other sampler
        if self.distributed_launch and not isinstance(dataset, IterableDataset):
            # sort or not
            if hasattr(self.hparams, "sorting"):
                shuffle_ddp = (
                    self.hparams.sorting == "random"
                )
            else:
                shuffle_ddp = True

            drop_last = loader_kwargs.get("drop_last", False)
            # num_replicas arg is equal to world_size
            # and retrieved automatically within
            # distributedSampler obj.
            if sampler is not None:
                sampler = DistributedSamplerWrapper(
                    sampler,
                    rank=self.rank,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = sampler
            elif loader_kwargs.get("batch_sampler") is None:
                # no sampler and batch-sampler
                train_sampler = DistributedSampler(
                    dataset,
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                    drop_last=drop_last,
                )

                # With DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = train_sampler
            else:
                sampler = DistributedSamplerWrapper(
                    loader_kwargs.get("batch_sampler", None),
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                )
                loader_kwargs["batch_sampler"] = sampler
        elif self.distributed_launch and isinstance(dataset, IterableDataset):
            logger.warning(
                "Cannot automatically solve distributed sampling "
                "for IterableDataset."
            )
        return loader_kwargs

    def on_fit_start(self):
        # Run this *after* starting all processes since jit module cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load dataset checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

    def init_optimizers(self):
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad(set_to_none)

    def on_evaluate_start(self, max_key=None, min_key=None):

        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(device_type=torch.device(self.device).type):
                outputs = self.compute_forward(batch, Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
        else:
            outputs = self.compute_forward(batch, Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, Stage.TRAIN)

            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        pass

    def check_gradients(self, loss):
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Print helpful debug info
            logger.warning(f"Loss is {loss}.")
            for p in self.modules.parameters():
                if not torch.isfinite(p).all():
                    logger.warning("Parameter is not finite: " + str(p))

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warning(
                    "Patience not yet exhausted, ignoring this batch."
                )
                return False

        # Clip gradient norm
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.modules.parameters()), self.max_grad_norm
            )

        return True

    def evaluate_batch(self, batch, stage):
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug model only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    if brain.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Profile only if desired (steps allow the profiler to know when all is is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                # Only run validation "on_stage_end" on main process
                self.step = 0
                run_on_main(
                    self.on_stage_end,
                    args=[Stage.VALID, avg_valid_loss, epoch],
                )

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs=None,
        valid_loader_kwargs=None
    ):
        if train_loader_kwargs is None:
            train_loader_kwargs = {}
        if valid_loader_kwargs is None:
            valid_loader_kwargs = {}

        if not isinstance(train_set, DataLoader):
            train_set = self.make_dataloader(
                train_set, stage=brain.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=brain.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        # Only show progressbar if requested and main_process
        enable = brain.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    @property
    def _optimizer_step_limit_exceeded(self):
        return (
            self.optimizer_step_limit is not None
            and self.optimizer_step >= self.optimizer_step_limit
        )

    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=1,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={INTRA_EPOCH_CKPT_FLAG: True},
            verbosity=logging.DEBUG,
        )

    def _compile_jit(self):
        """Compile requested modules with ``torch.jit.script``."""
        if self.jit_module_keys is None:
            return

        for name in self.jit_module_keys:
            if name not in self.modules:
                raise ValueError(
                    "module" + name + " is not defined in your hparams file."
                )
            module = torch.jit.script(self.modules[name])
            self.modules[name] = module.to(self.device)

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if self.distributed_launch:
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    if self.distributed_backend == "gloo":
                        module = DDP(
                            module,
                            device_ids=None,
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    else:
                        module = DDP(
                            module,
                            device_ids=[self.device],
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    self.modules[name] = module

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs=None,
    ):
        if test_loader_kwargs is None:
            test_loader_kwargs = {}
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0
        return avg_test_loss

    def update_average(self, loss, avg_loss):
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

    @contextmanager
    def no_sync(self, use=True):
        """Copies pytorch's implementation for doing no_sync across all modules.

        Explanation: nn.module.no_sync() is a context manager for when one does
        not want to sync gradients, which happens when using bot DDP and gradient accumulation.
        brain's class can contain multiple modules and calling no_sync no these
        individually would be very awkward, therefore this contextmanager exists.

        Parameters
        ----------
        use : bool
            If set to `False` will still sync gradients, useful to make behaviour togglable.

        Returns
        -------

        """
        if use:
            old_values_list = []
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    # if not using DDP
                    break
                old_values_list.append(module.require_backward_grad_sync)
                module.require_backward_grad_sync = False
            yield
            for module, old_value in zip(
                self.modules.values(), old_values_list
            ):
                if not hasattr(module, "require_backward_grad_sync"):
                    break
                module.require_backward_grad_sync = old_value
        else:
            yield

    @brain.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict))

    @brain.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        with open(path) as f:
            save_dict = yaml.safe_load(f)
        self.step = save_dict["step"]
        self.avg_train_loss = save_dict["avg_train_loss"]
        # Ensure compatibility with checkpoints from before optimizer_step:
        if "optimizer_step" not in save_dict:
            clsname = self.__class__.__name__
            MSG = f"'optimizer_step' not found in {clsname} checkpoint."
            MSG += " Using the saved 'step' value (BACKWARDS COMPATIBILITY)"
            warnings.warn(MSG)
            self.optimizer_step = self.step
        else:
            self.optimizer_step = save_dict["optimizer_step"]
