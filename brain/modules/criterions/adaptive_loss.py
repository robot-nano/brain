class AdaptiveLoss:

    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
