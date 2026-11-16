import math
import torch
import torch.nn as nn


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = math.gcd(num_channels, max_groups)
    groups = max(1, groups)
    return nn.GroupNorm(groups, num_channels)


class FrequencyHighEnhancement(nn.Module):
    def __init__(self, dim: int, cutoff: float = 0.18, temperature: float = 18.0):
        super().__init__()
        self.temperature = temperature

        # cutoff 限制在 [0.05, 0.45]，初始值约为 cutoff
        self.cutoff_min = 0.05
        self.cutoff_max = 0.45

        init = (cutoff - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        init = max(1e-4, min(1 - 1e-4, init))
        self.cutoff_logit = nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))

        self.norm = _make_gn(dim)
        self.high_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            _make_gn(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def get_cutoff(self):
        return self.cutoff_min + (self.cutoff_max - self.cutoff_min) * torch.sigmoid(self.cutoff_logit)

    def _high_pass_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        fy = torch.fft.fftfreq(h, d=1.0, device=device).view(1, 1, h, 1)
        fx = torch.fft.rfftfreq(w, d=1.0, device=device).view(1, 1, 1, w // 2 + 1)
        radius = torch.sqrt(fx * fx + fy * fy) / math.sqrt(0.5)

        cutoff = self.get_cutoff().to(device=device)

        return torch.sigmoid((radius - cutoff) * self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        b, c, h, w = x.shape

        xf = x.float()
        x_norm = self.norm(xf)

        freq = torch.fft.rfft2(x_norm, norm="ortho")
        mask = self._high_pass_mask(h, w, x.device)

        high = torch.fft.irfft2(freq * mask, s=(h, w), norm="ortho")
        high = self.high_proj(high).to(dtype=dtype)

        gate = self.gate(xf).to(dtype=dtype)
        gamma = self.gamma.to(dtype=dtype)

        return x + gamma * gate * high


class CrackCandidateSparseAttention(nn.Module):
    def __init__(self, dim: int, candidate_ratio: float = 0.125, min_tokens: int = 64,
                 max_tokens: int = 1024, max_heads: int = 4, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.candidate_ratio = candidate_ratio
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        hidden = max(dim // 4, 4)
        self.score = nn.Sequential(
            _make_gn(dim),
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        num_heads = min(max_heads, dim)
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))
        self.out_gate = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=True), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _num_candidates(self, n_tokens: int) -> int:
        k = int(n_tokens * self.candidate_ratio)
        k = max(self.min_tokens, k)
        k = min(self.max_tokens, k, n_tokens)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        k = self._num_candidates(n)
        tokens = x.flatten(2).transpose(1, 2)
        scores = self.score(x.float()).flatten(2).squeeze(1)
        idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, c)
        cand = torch.gather(tokens, dim=1, index=idx_expand)
        cand_norm = self.norm(cand.float()).to(dtype=cand.dtype)
        attn_out, _ = self.attn(cand_norm, cand_norm, cand_norm, need_weights=False)
        attn_out = self.proj(attn_out).to(dtype=tokens.dtype)
        sparse_tokens = torch.zeros_like(tokens)
        sparse_tokens.scatter_(dim=1, index=idx_expand, src=attn_out)
        sparse_map = sparse_tokens.transpose(1, 2).reshape(b, c, h, w)
        gate = self.out_gate(x.float()).to(dtype=x.dtype)
        gamma = self.gamma.to(dtype=x.dtype)
        return x + gamma * gate * sparse_map
