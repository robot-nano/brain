import torch
import brain


class FeatsFwd(torch.nn.Module):
    def __init__(
        self,
        modules,
    ):
        super().__init__()
        self.fbkans = modules[0]
        self.normalize = modules[1]
        self.cnn = modules[2]
        self.transformer = modules[3]

    def forward(self, wav):
        feats = self.fbkans(wav)
        feats = self.normalize(feats, torch.tensor([1.]), epoch=5)

        in_feats = feats.unsqueeze(1)
        src = self.cnn(in_feats)
        in_src = src.transpose(1, 2)
        out = self.transformer.encode(in_src)
        return out


class Decode(torch.nn.Module):
    def __init__(self, transformer, seq_lin, softmax):
        super().__init__()
        self.transformer = transformer
        self.fc = seq_lin
        self.softmax = softmax

    def forward(self, tgt, enc_out):
        out, _ = self.transformer.decode(tgt, enc_out)
        prob_list = self.softmax(self.fc(out))
        return prob_list
