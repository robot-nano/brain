class LabelSmoothedCrossEntropyWithCtcCriterion(object):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        ctc_weight,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.ctc_weight = ctc_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.0).type_as(loss)
        if self.ctc_weight > 0.0:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"
            ctc_loss = (
                F.ctc_loss(
                    ctc_lprobs,
                    ctc_tgt_flat,
                    ctc_lens,
                    ctc_tgt_lens,
                    reduction=reduction,
                    zero_infinity=True,
                )
                * self.ctc_weight
            )
        loss += ctc_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output