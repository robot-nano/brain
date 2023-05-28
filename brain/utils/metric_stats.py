import torch
from joblib import Parallel, delayed
from brain.utils.data_utils import undo_padding
from brain.utils.edit_distance import wer_summary, wer_details_for_batch
from brain.dataio.dataio import (
    merge_char,
    split_word,
    extract_concepts_values,
)
from brain.dataio.wer import print_wer_summary, print_alignments


class MetricStats:
    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, *args, **kwargs):
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.metric(*args, **kwargs).detach()

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores = sequence_evaluation(metric=self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

        self.scores.extend(scores)

    def summarize(self, field=None):
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "average": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        if not self.summary:
            self.summarize()

        message = f"Average score: {self.summary['average']}\n"
        message += f"Min error: {self.summary['min_score']}\n"
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']}\n"
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


def multiprocess_evaluation(metric, predict, target, lengths=None, n_jobs=8):
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    while True:
        try:
            scores = Parallel(n_jobs=n_jobs, timeout=30)(
                delayed(metric)(p, t) for p, t in zip(predict, target)
            )
            break
        except Exception as e:
            print(e)
            print("Evaluation timeout...... (will try again)")

    return scores


def sequence_evaluation(metric, predict, target, lengths=None):
    """Runs metric evaluation sequentially over the inputs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    scores = []
    for p, t in zip(predict, target):
        score = metric(p, t)
        scores.append(score)
    return scores


class ErrorRateStats(MetricStats):
    def __init__(
        self,
        merge_tokens=False,
        split_tokens=False,
        space_token="_",
        keep_values=True,
        extract_concepts_values=False,
        tag_in="",
        tag_out="",
    ):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token
        self.extract_concepts_values = extract_concepts_values
        self.keep_values = keep_values
        self.tag_in = tag_in
        self.tag_out = tag_out

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        if self.extract_concepts_values:
            predict = extract_concepts_values(
                predict,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )
            target = extract_concepts_values(
                target,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )

        scores = wer_details_for_batch(ids, target, predict, True)

        self.scores.extend(scores)

    def summarize(self, field=None):
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)
