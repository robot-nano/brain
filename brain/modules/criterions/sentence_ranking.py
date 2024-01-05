class SentenceRankingCriterion(object):
    def __init__(self, task, ranking_head_name, save_predictions, num_classes):
        super().__init__(task)
        self.ranking_head_name = ranking_head_name
        if save_predictions is not None:
            self.prediction_h = open(save_predictions, "w")
        else:
            self.prediction_h = None
        self.num_classes = num_classes

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--ranking-head-name',
                            default='sentence_classification_head',
                            help='name of the ranking head to use')

    def forward(self, model, sample, reduce=True):

        scores = []
        for idx in range(self.num_classes):
            score, _ = model(
                **sample["net_input{idx}".format(idx=idx + 1)],
                classification_head_name=self.ranking_head_name,
            )
            scores.append(score)

        logits = torch.cat(scores, dim=1)
        sample_size = logits.size(0)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )