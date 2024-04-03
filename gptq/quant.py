from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(x, scale, zero, maxq, eps=1e-9):
    q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)
    return q


def dequantize(q, scale, zero):
    return scale * (q.to(scale.dtype) - zero.to(scale.dtype))


def quant_proj(x, scale, zero, maxq, eps=1e-9):
    return dequantize(quantize(x, scale, zero, maxq, eps), scale, zero)


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, sym=True, grid=100, maxshrink=0.75):
        self.maxq = torch.tensor(2**bits - 1)
        self.sym = sym
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, solve=None, scales=None):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        x = x.flatten(1)
        if scales is not None:
            x *= scales

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if solve is not None:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid) + 1):
                p = 1 - i / self.grid
                clip = p * torch.max(xmax, torch.abs(xmin))
                xmax1 = torch.min(xmax, +clip)
                xmin1 = torch.max(xmin, -clip)
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quant_proj(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                if scales is not None:
                    q /= scales
                delta = q - x
                err = torch.sum(torch.linalg.solve_triangular(solve, delta, upper=True, left=False) ** 2, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x):
        return quantize(x, self.scale, self.zero, self.maxq)

    def dequantize(self, x):
        return dequantize(x, self.scale, self.zero)


class QLinear(nn.Module):
    """
    Temporary wrapper for model finetuning.
    """

    def __init__(
        self, qweight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.in_features = qweight.shape[1]
        self.out_features = qweight.shape[0]
        self.num_groups = scale.shape[1]

        qweight = qweight.view(self.out_features, self.num_groups, -1)
        scale = scale.view(self.out_features, -1, 1)
        zero = zero.view(self.out_features, -1, 1)
        self.register_buffer("qweight", qweight)
        self.scale = nn.Parameter(scale)
        self.register_buffer("zero", zero)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def get_weight(self):
        return dequantize(self.qweight, self.scale, self.zero).view(self.out_features, self.in_features)

    def forward(self, input: torch.Tensor):
        # get weight without outliers
        weight = self.get_weight()
        out = F.linear(input, weight, self.bias)
        return out
