import torch

from .seq2seq import S2SBaseSearcher


class S2SBeamSearcher(S2SBaseSearcher):
    def __init__(
        self,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        length_rewarding=0,
        coverage_penalty=0.0,
        lm_weight=0.0,
        lm_modules=None,
        ctc_weight=0.0,
        blank_index=0,
        using_max_attn_shift=False,
        minus_inf=-1e20,
    ):
        super().__init__(
            bos_index, eos_index, min_decode_ratio, max_decode_ratio
        )
        self.beam_size = beam_size
        self.topk = topk
        self.return_log_probs = return_log_probs
        self.length_normalization = length_normalization
        self.length_rewarding = length_rewarding
        self.coverage_penalty = coverage_penalty
        self.coverage = None

        if self.length_normalization and self.length_rewarding > 0:
            raise ValueError(
                "length normalization is not compatible with length rewarding."
            )

        self.using_eos_threshold = using_eos_threshold
        self.eos_threshold = eos_threshold
        self.lm_weight = lm_weight
        self.lm_modules = lm_modules

        self.minus_inf = minus_inf
        self.blank_index = blank_index

        self.minus_inf = minus_inf

    def _check_full_beams(self, hyps, beam_size):
        hyps_len = [len(lst) for lst in hyps]
        beam_size = [self.beam_size for _ in range(len(hyps_len))]
        if hyps_len == beam_size:
            return True
        else:
            return False

    def _update_hyp_and_scores(
        self,
        inp_tokens,
        alived_seq,
        alived_log_probs,
        hyps_and_scores,
        scores,
        timesteps,
    ):
        is_eos = inp_tokens.eq(self.eos_index)
        (eos_indices,) = torch.nonzero(is_eos, as_tuple=True)

        if eos_indices.shape[0] > 0:
            for index in eos_indices:
                index = index.item()
                batch_id = torch.div(
                    index, self.beam_size, rounding_mode="floor"
                )
                if len(hyps_and_scores[batch_id]) == self.beam_size:
                    continue
                hyp = alived_seq[index, :]
                log_probs = alived_log_probs[index, :]
                final_scores = scores[index] + self.length_rewarding * (
                    timesteps + 1
                )
                hyps_and_scores[batch_id].append((hyp, log_probs, final_scores))
        return is_eos

    def forward(self, enc_states, wav_len):
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size * self.beam_size, device=device)

        if self.lm_weight > 0:
            lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)

        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

        inp_tokens = (
            torch.zeros(batch_size * self.beam_size, device=device)
            .fill_(self.bos_index)
            .long()
        )

        self.beam_offset = (
            torch.arange(batch_size, device=device) * self.beam_size
        )

        sequence_scores = torch.empty(
            batch_size * self.beam_size, device=device
        )
        sequence_scores.fill_(float("-inf"))

        sequence_scores.index_fill_(0, self.beam_offset, 0.0)

        hyps_and_scores = [[] for _ in range(batch_size)]

        alived_seq = torch.empty(
            batch_size * self.beam_size, 0, device=device
        ).long()

        alived_log_probs = torch.empty(
            batch_size * self.beam_size, 0, device=device
        )

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        min_decode_steps, max_decode_steps = self.change_max_decoding_length(
            min_decode_steps, max_decode_steps
        )

        prev_attn_peak = torch.zeros(batch_size * self.beam_size, device=device)

        for t in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(hyps_and_scores, self.beam_size):
                break

            log_probs, memory, attn = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )

            log_probs_clone = log_probs.clone().reshape(batch_size, -1)
            vocab_size = log_probs.shape[-1]

            if t < min_decode_steps:
                log_probs[:, self.eos_index] = self.minus

            if self.lm_weight > 0:
                lm_log_probs, lm_memory = self.lm_forward_step(
                    inp_tokens, lm_memory
                )
                log_probs = log_probs + self.lm_weight * lm_log_probs

            scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
            scores = scores + log_probs

            if self.length_normalization:
                scores = scores / (t + 1)

            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            inp_tokens = (candidates % vocab_size).view(
                batch_size * self.beam_size
            )

            scores = scores.view(batch_size * self.beam_size)
            sequence_scores = scores

            if self.length_normalization:
                sequence_scores = sequence_scores * (t + 1)

            predecessors = (
                torch.div(candidates, vocab_size, rounding_mode="floor")
                + self.beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(batch_size * self.beam_size)

            memory = self.permute_mem(memory, index=predecessors)
            if self.lm_weight > 0:
                lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)

            alived_seq = torch.cat(
                [
                    torch.index_select(alived_seq, dim=0, index=predecessors),
                    inp_tokens.unsqueeze(1),
                ],
                dim=-1,
            )

            beam_log_probs = log_probs_clone[
                torch.arange(batch_size).unsqueeze(1), candidates
            ].reshape(batch_size * self.beam_size)
            alived_log_probs = torch.cat(
                [
                    torch.index_select(
                        alived_log_probs, dim=0, index=predecessors
                    ),
                    beam_log_probs.unsqueeze(1),
                ],
                dim=-1,
            )

            is_eos = self._update_hyp_and_scores(
                inp_tokens,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=t,
            )

            sequence_scores.masked_fill_(is_eos, float("-inf"))


class S2STransformerBeamSearch(S2SBaseSearcher):
    def __init__(
        self, modules, temperature=1.0, temperature_lm=1.0, **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm

    def reset_mem(self, batch_size, device):
        return None

    def reset_lm_mem(self, batch_size, device):
        return None

    def permute_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        memory = _update_mem(inp_tokens, memory)
        pred, attn = self.model.decode(memory, enc_states)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn


def inflate_tensor(tensor, times, dim):
    return torch.repeat_interleave(tensor, times, dim=dim)


def _update_mem(inp_tokens, memory):
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)
