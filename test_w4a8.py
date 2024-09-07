import unittest

import numpy as np
import torch
import torch.nn as nn

import marlin


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')


def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    s_extra = ref.t().abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float)
    s_extra = s_extra.reshape(1, n)
    fake_quant_ref = (ref / s_extra).round().clamp(-128, 127).to(torch.int8)
    fake_quant_ref = (fake_quant_ref * s_extra.float()).half()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.W4A8Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s_group = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.s_channel = torch.empty((1, n), dtype=torch.float, device=DEV)
    if groupsize == m:
        layer.pack(linear, s.t())
    else:
        layer.pack(linear, s.t(), s_extra)
    q = layer.B
    s2 = layer.s_channel
    s3 = layer.s_group
    return ref, q, fake_quant_ref, s2, s3



class Test(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        if k == groupsize:
            groupsize = -1
        A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=DEV)
        B_ref, B, fake_quant_B, s2, s3 = gen_quant4(k, n, groupsize=groupsize)
        s1_ref = torch.rand((m, 1), dtype=torch.float, device=DEV)
        s1 = s1_ref
        max_par = 16
        C = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=DEV)
        D = torch.zeros((m, n), dtype=torch.half, device=DEV)
        if groupsize == -1:
            D_ref = torch.matmul(A.half() * s1_ref.half(), B_ref)
        else:
            D_ref = torch.matmul(A.half() * s1_ref.half(), fake_quant_B)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin.w4a8_mul(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, -1, max_par=max_par)
        torch.cuda.synchronize()
        self.assertLess(torch.mean(torch.abs(D - D_ref)) / torch.mean(torch.abs(D_ref)), 0.003)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(128, 128)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0], thread_k, thread_n, 128)

    def test_errors(self):
        print()
        m, n, k = 16, 256, 64
        A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=DEV)
        B_ref, B, fake_quant_B, s2, s3 = gen_quant4(k, n)
        s1 = torch.rand((m, 1), dtype=torch.float, device=DEV)
        s3 = torch.tensor([], dtype=torch.half, device=DEV)
        max_par = 16
        C = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=DEV)
        D = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            marlin.w4a8_mul(A, B, C, D, s1, s2, s3, workspace, 128, 128, -1, max_par=max_par)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            marlin.w4a8_mul(A, B, C, D, s1, s2, s3, workspace, 256, 256, -1, max_par=max_par)
        except:
            err = True 
        self.assertTrue(err)
        s1 = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            marlin.w4a8_mul(A, B, C, D, s1, s2, s3, workspace, 256, 256, -1, max_par=max_par)
        except:
            err = True 
        self.assertTrue(err)

    def test_groups(self):
        print()
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)


if __name__ == '__main__':
    unittest.main()