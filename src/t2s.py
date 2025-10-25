import torch
from functools import reduce
from scipy.stats import norm

class T2SMark:
    def __init__(self, m, tau, latent_shape):
        self.latent_shape = latent_shape # [C, H, W]
        self.n = reduce(lambda x, y: x * y, self.latent_shape, 1)
        self.m = m
        self.r = int(2 * norm.cdf(-tau) * self.n / m)
        self.k = self.m * self.r
        self.noise_size = self.n - self.k
        self.prng = torch.Generator()

    def binlist2int(self, binlist):
        res = reduce(lambda x, y: x * 2 + y, binlist)
        if isinstance(binlist, torch.Tensor):
            return res.item()
        return res

    def encode(self, b, K):
        z = torch.randn(self.latent_shape).cuda().flatten()
        self.prng.manual_seed(self.binlist2int(K))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng).cuda().float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng)[:self.k]

        b_r = (1 - 2 * b).repeat(self.r).cuda().float()
        codeword = b_r * v_value

        w = torch.zeros(self.n).bool()
        w[v_support] = True

        tail = torch.topk(z.abs(), k=self.k, dim=0, largest=True, sorted=False)
        central = torch.topk(z.abs(), k=self.noise_size, dim=0, largest=False, sorted=False)

        z_w = torch.zeros(self.n).cuda()
        z_w[w] = tail.values * codeword
        z_w[~w] = central.values * (torch.randint(0, 2, (self.noise_size,)).float() * 2 - 1).cuda()
        return z_w.reshape(self.latent_shape)

    def decode(self, reversed_noise, K, detection=False):
        self.prng.manual_seed(self.binlist2int(K))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng).cuda().float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng)[:self.k]

        w = torch.zeros(self.n).bool()
        w[v_support] = True

        watermarked_vec = (reversed_noise.flatten()[w] * v_value).cuda()
        p = watermarked_vec.reshape(self.r, self.m).sum(dim=0)
        b = (p < 0).int()
        if detection:
            return b, torch.norm(p.flatten(), p=1).item()
        return b