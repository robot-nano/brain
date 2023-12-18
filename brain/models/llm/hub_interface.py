class BARTHub(object):
    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]
    
    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        pass

    def decode(self, tokens: torch.LongTensor):
        pass

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample