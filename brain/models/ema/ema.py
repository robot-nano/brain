class EMA(object):
    def __init__(self, model, config, device=None, skip_keys=None):

        self.decay = config.ema_decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.config = config
        self.skip_keys = skip_keys or set()
        self.fp32_params = {}

    def get_model(self):
        return self.model

    def build_fp32_params(self, state_dict=None):
        if not self.config.ema_fp32:
            raise RuntimeError(
                "build_fp32_params should not be called if ema_fp32=False. "
                "Use ema_fp32=True if this is really intended."
            )

        if state_dict is None:
            state_dict = self.model.state_dict()

        def _to_float(t):
            return t.float() if torch.is_floating_point(t) else t

        for param_key in state_dict:
            if param_key in self.fp32_params:
                self.fp32_params[param_key].copy_(state_dict[param_key])
            else:
                self.fp32_params[param_key] = _to_float(state_dict[param_key])

    def setp(self, new_model, updates=None):
         if updates is not None:
            self._set_decay(
                0 if updates < self.config.ema_start_update else self.config.ema_decay
            )
        if self.config.ema_update_freq > 1:
            self.update_freq_counter += 1
            if self.update_freq_counter >= self.config.ema_update_freq:
                self._step_internal(new_model, updates)
                self.update_freq_counter = 0
        else:
            self._step_internal(new_model, updates)