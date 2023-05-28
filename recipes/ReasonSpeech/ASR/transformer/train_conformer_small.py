import sys
import random
import copy
import numpy as np
import torch
import torchaudio
import brain
from brain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import pdb


class ASR(brain.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        in_feats = feats.unsqueeze(1)
        src = self.modules.CNN(in_feats)
        in_src = src.transpose(1, 2)
        enc_out, pred = self.modules.Transformer(
            in_src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # compute outputs
        hyps = None
        if stage == brain.Stage.TRAIN:
            hyps = None
        elif stage == brain.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == brain.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        (p_seq, wav_lens, hyps) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        if stage != brain.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == brain.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq) for utt_seq in hyps
                ]
                target_words = batch.wrd
                self.cer_metric.append(ids, predicted_words, target_words)

            # compute_ the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, brain.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, brain.Stage.TRAIN)
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
                self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, brain.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, brain.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        if stage != brain.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_status = {"loss": stage_loss}
        if stage == brain.Stage.TRAIN:
            self.train_stats = stage_status
        else:
            stage_status["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == brain.Stage.TEST
            ):
                stage_status["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == brain.Stage.VALID and brain.utils.distributed.if_main_process():
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_status,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_status["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=3,
            )
        elif stage == brain.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_status,
            )
            with open(self.hparams.cer_file, 'w') as w:
                self.cer_metric.write_stats(w)

            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )


def dataio_prepare(hparams):
    train_data = brain.dataio.dataset.DynamicItemDataset.from_csv(hparams["train_csv"])
    valid_data = brain.dataio.dataset.DynamicItemDataset.from_csv(hparams["valid_csv"])
    test_data = brain.dataio.dataset.DynamicItemDataset.from_csv(hparams["test_csv"])

    datasets = [train_data, valid_data, test_data]

    @brain.dataio.data_pipeline.takes("wav")
    @brain.dataio.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig, rate = torchaudio.load(wav)
        resampled = torchaudio.transforms.Resample(
            rate, hparams["sample_rate"],
        )(sig.squeeze(0))
        yield resampled

    brain.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    tokenizer = hparams["tokenizer"]

    @brain.dataio.data_pipeline.takes("wrd")
    @brain.dataio.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        yield tokens_list
        replace_len = int(len(tokens_list) * hparams["noise_percent"])
        if replace_len > 0:
            re_val = [random.randint(2, hparams['output_neurons'] - 51) for _ in range(replace_len)]
            re_index = random.sample(range(len(tokens_list)), replace_len)
            tokens_bos = np.array(copy.deepcopy(tokens_list))
            tokens_bos[re_index] = re_val
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + list(tokens_bos))
        else:
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + list(tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    brain.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    brain.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
    )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer
    )


if __name__ == "__main__":
    hparams_file, run_opts, overrides = brain.parse_arguments(sys.argv[1:])
    brain.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    brain.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    (
        train_set,
        valid_set,
        test_set,
        tokenizer,
    ) = dataio_prepare(hparams)

    # load pretrained tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["train_dataloader_opts"],
    )
    print(" ")
