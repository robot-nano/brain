class Adamax(torch.optim.Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm).

    It has been proposed in `Adam: A Method for Stochastic Optimization`__.

    Compared to the version in PyTorch, this version implements a fix for weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        bias_correction (bool, optional): enable bias correction (default: True)

    __ https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        bias_correction=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super(Adamax, self).__init__(params, defaults)