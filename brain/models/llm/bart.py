class BARTModel(object):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()