import math
import time
from typing import Tuple

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer, stable=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.stable = stable
        self.mean = torch.zeros((self.columns, 1), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.stable:
            inp = inp.float()
            delta = torch.mean(inp, 1, keepdims=True) - self.mean
            self.H += inp.matmul(inp.t()) + delta.matmul(delta.t()) * self.nsamples * tmp / (self.nsamples + tmp)
            self.nsamples += tmp
            self.mean += delta * tmp / self.nsamples
        else:
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.1, groupsize=-1, clip=False, baseline=False) -> Tuple[torch.Tensor]:
        dtype = self.layer.weight.dtype
        W = self.layer.weight.data.clone()
        W = W.float()

        rows, cols = W.shape
        if groupsize == -1:
            groupsize = cols
        num_groups = cols // groupsize

        tick = time.perf_counter()

        if self.stable:
            self.H /= self.nsamples
            self.H += self.mean.matmul(self.mean.t())
            self.H *= 2
        H = self.H
        del self.H

        Losses = torch.zeros_like(W)
        # quantized weight is a tuple of qweight, scale and zero
        qweight = torch.empty_like(W, dtype=torch.uint8)
        scale = torch.empty(rows, num_groups, dtype=dtype, device=self.dev)
        zero = torch.empty(rows, num_groups, dtype=torch.uint8, device=self.dev)

        if not baseline:
            try:
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
            except:
                print("Singularity.")
                baseline = True
        if baseline:
            del H
            Hinv = torch.eye(self.columns, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if (i + i1) % groupsize == 0:
                    self.quantizer.find_params(W1, solve=Hinv1 if clip else None)
                    scale[:, (i + i1) // groupsize] = self.quantizer.scale.flatten()
                    zero[:, (i + i1) // groupsize] = self.quantizer.zero.flatten()

                q = self.quantizer.quantize(w.unsqueeze(1))
                w_q = self.quantizer.dequantize(q).flatten()

                qweight[:, i1 + i] = q.flatten()
                Losses1[:, i] = (w - w_q) ** 2 / d**2

                err1 = (w - w_q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print(f"Time: {(time.perf_counter() - tick):.2f}")
        print(f"L2 Loss: {torch.sum(Losses).item():.2f}")

        return qweight, scale.to(dtype), zero.to(dtype)
