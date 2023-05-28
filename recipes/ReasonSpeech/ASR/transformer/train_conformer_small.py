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
        wav, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        #
        pdb.set_trace()
        print(" ")

    def compute_objectives(self, predictions, batch, stage):
        return 1


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
        checkpointer=None,
    )

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_set,
        train_loader_kwargs=hparams["train_dataloader_opts"],
    )
    print(" ")
