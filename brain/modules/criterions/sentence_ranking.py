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