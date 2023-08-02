import torch


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class SoftmaxOne(torch.nn.Module):
    def __init__(self, dim=None, _stacklevel=3, dtype=None):
        super().__init__()

        self.dim = dim
        self._stacklevel = _stacklevel
        self.dtype = dtype

    def forward(self, x):
        return softmax_one(
            x,
            dim=self.dim,
            _stacklevel=self._stacklevel,
            dtype=self.dtype,
        )
