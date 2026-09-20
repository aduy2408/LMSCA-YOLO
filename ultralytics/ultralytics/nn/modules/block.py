# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_related.ultralytics.ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import LayerNorm2d, TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BiLevelRoutingAttention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fKV",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "DeformableHeadConv",
    "EnSimAM",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "IRDCB",
    "KVCompressedAttention",
    "KVCompressedAttentionPartial",
    "KVCompressedTransformerEncoder",
    "LearnableEnSimAM",
    "LDown",
    "P4CrossScaleLocalAttention",
    "SAGRI",
    "SAGRIConvDown",
    "P2PartialKVToP3",
    "Proto",
    "RegionRoutingAttentionLite",
    "TopKAdaptiveGroupKVAttention",
    "TopKGlobalGroupKVAttention",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class _GridSampleDeformConv2d(nn.Module):
    """Small pure-PyTorch deformable conv used to avoid torchvision native CUDA crashes."""

    def __init__(self, c1: int, c2: int, k: int = 3, padding: int = 1):
        super().__init__()
        self.k = k
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        offset = offset.float()
        b, _, h, w = x.shape
        k2 = self.k * self.k
        if offset.shape[1] != 2 * k2:
            raise ValueError(f"Expected {2 * k2} offset channels, got {offset.shape[1]}")

        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        hp, wp = x.shape[-2:]

        device = x.device
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        yy = yy.unsqueeze(0)
        xx = xx.unsqueeze(0)

        out = x.new_zeros(b, self.weight.shape[0], h, w)
        for ky in range(self.k):
            for kx in range(self.k):
                idx = ky * self.k + kx
                sample_y = yy + ky + offset[:, 2 * idx]
                sample_x = xx + kx + offset[:, 2 * idx + 1]
                grid_y = sample_y.mul(2.0 / max(hp - 1, 1)).sub(1.0)
                grid_x = sample_x.mul(2.0 / max(wp - 1, 1)).sub(1.0)
                grid = torch.stack((grid_x, grid_y), dim=-1)
                sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                out = out + torch.einsum("bchw,oc->bohw", sampled, self.weight[:, :, ky, kx].float())
        return out.to(orig_dtype)


class DeformableHeadConv(nn.Module):
    """Lightweight deformable refinement block for detection-head localization features."""

    def __init__(self, c1: int, c2: int, k: int = 3, gamma_init: float = 0.0, act: bool = True):
        """Initialize a DCNv2-style head block.

        The offset path is zero-initialized so the block starts close to a regular head refinement layer.
        """
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"DeformableHeadConv kernel size must be odd, got {k}")
        self.offset = nn.Conv2d(c1, 2 * k * k, kernel_size=3, stride=1, padding=1)
        self.deform = _GridSampleDeformConv2d(c1, c2, k=k, padding=autopad(k))
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act else nn.Identity()
        self.shortcut = c1 == c2
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))) if self.shortcut else None
        self.proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deformable refinement and optional residual gating."""
        y = self.deform(x, self.offset(x))
        y = self.act(self.bn(y))
        if self.shortcut:
            return x + self.gamma * y
        return self.proj(x) + y


class EnSimAM(nn.Module):
    """Enhanced SimAM attention with global, local-variance, and edge-response branches."""

    def __init__(self, lambd: float = 1e-4, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        """Initialize parameter-free attention coefficients and fixed Sobel kernels."""
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free attention without changing tensor shape."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
        a_global = torch.sigmoid(1.0 / energy)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        a_local = torch.sigmoid(self.alpha * local_var)

        c = x.shape[1]
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=c)
        gy = F.conv2d(x, sobel_y, padding=1, groups=c)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        a_edge = torch.sigmoid(self.beta * edge)

        attention = 0.5 * a_global + 0.25 * a_local + 0.25 * a_edge
        return x * attention


class LearnableEnSimAM(nn.Module):
    """Enhanced SimAM with learnable global/local/edge mixing and residual gating."""

    def __init__(
        self,
        lambd: float = 1e-4,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-6,
        gamma_init: float = 1.0,
    ):
        """Initialize learnable branch weights from the fixed EnSimAM prior."""
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.branch_logits = nn.Parameter(torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable branch mixing without changing tensor shape."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
        a_global = torch.sigmoid(1.0 / energy)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        a_local = torch.sigmoid(self.alpha * local_var)

        c = x.shape[1]
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=c)
        gy = F.conv2d(x, sobel_y, padding=1, groups=c)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        a_edge = torch.sigmoid(self.beta * edge)

        weights = self.branch_logits.softmax(dim=0).to(dtype=x.dtype, device=x.device)
        attention = weights[0] * a_global + weights[1] * a_local + weights[2] * a_edge
        attended = x * attention
        return x + self.gamma * (attended - x)



def _choose_attention_heads(channels: int, requested_heads: int) -> int:
    """Pick a valid attention head count that divides channels."""
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class KVCompressedAttention(nn.Module):
    """Self-attention with full-resolution queries and spatially compressed keys/values.

    Supports multiple K/V spatial compression modes:
    - avg: AvgPool + GroupNorm for K and V.
    - avg_dwk/dwconv: AvgPool + DWConv + GroupNorm for K, AvgPool + GroupNorm for V.
    - dw_stride: stride depthwise Conv + GroupNorm for K and V.
    - group_weight: learned softmax weighting inside each sr_ratio x sr_ratio group.
    Attention is computed via ``F.scaled_dot_product_attention`` when available.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        residual: bool = True,
    ):
        """Initialize KV-compressed attention.

        Args:
            c1: Input channels.
            c2: Output channels.
            num_heads: Requested attention heads. Reduced if it does not divide c2.
            sr_ratio: Spatial compression ratio for K/V tokens.
            mode: ``avg``, ``avg_dwk``, ``dw_stride``, ``dwconv``, or ``group_weight`` compression.
            attn_drop: Dropout probability applied to attention weights during training.
            residual: Whether to add the projected attention output back to the input projection.
        """
        super().__init__()
        if mode == "dwconv":
            mode = "avg_dwk"
        if mode not in {"avg", "avg_dwk", "dw_stride", "group_weight"}:
            raise ValueError(f"Unsupported KV compression mode: {mode}")

        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = max(1, int(sr_ratio))
        self.mode = mode
        self.attn_drop_p = attn_drop
        self.residual = residual

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)

        if self.sr_ratio > 1 and self.mode == "avg":
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "avg_dwk":
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "dw_stride":
            self.k_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
        else:
            self.k_compress = nn.Identity()
            self.v_compress = nn.Identity()

        self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)

        self.group_score = nn.Linear(c2, 1) if self.mode == "group_weight" and self.sr_ratio > 1 else None
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False) if self.mode == "group_weight" else None

        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    def _compress_group_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Compress each sr_ratio x sr_ratio token group with learned softmax weights."""
        if self.sr_ratio <= 1:
            return x
        b, c, h, w = x.shape
        pad_h = (self.sr_ratio - h % self.sr_ratio) % self.sr_ratio
        pad_w = (self.sr_ratio - w % self.sr_ratio) % self.sr_ratio
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        gh, gw = hp // self.sr_ratio, wp // self.sr_ratio
        tokens = x.view(b, c, gh, self.sr_ratio, gw, self.sr_ratio).permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = tokens.view(b, gh, gw, self.sr_ratio * self.sr_ratio, c)
        weights = self.group_score(tokens).softmax(dim=3)
        compressed = (tokens * weights).sum(dim=3)
        return compressed.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KV-compressed attention and return a BCHW tensor."""
        x = self.input_proj(x)
        b, c, h, w = x.shape

        q = self.q(x).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        if self.mode == "group_weight" and self.group_score is not None:
            kv_source = self._compress_group_weight(x)
            kv = self.kv(kv_source).flatten(2).transpose(1, 2)
            kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            k_src = self.k_compress(x)
            v_src = self.v_compress(x)
            k = self.k_proj(k_src)
            v = self.v_proj(v_src)

            def _to_heads(t: torch.Tensor) -> torch.Tensor:
                n = t.shape[2] * t.shape[3]
                return t.flatten(2).transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k = _to_heads(k)
            v = _to_heads(v)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj_bn(self.proj(out))
        return x + out if self.residual else out


class KVCompressedTransformerEncoder(nn.Module):
    """Pre-norm transformer encoder block using KV-compressed self-attention and a DW-PW convolutional FFN."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        """Initialize LayerNorm-KVCA and LayerNorm-DW-FFN residual branches."""
        super().__init__()
        hidden = max(c2, int(c2 * mlp_ratio))
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.norm1 = LayerNorm2d(c2)
        self.attn = KVCompressedAttention(c2, c2, num_heads, sr_ratio, mode, attn_drop=attn_drop, residual=False)
        self.norm2 = LayerNorm2d(c2)
        self.ffn = nn.Sequential(
            Conv(c2, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, c2, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LN -> KVCA -> residual, then LN -> DW-FFN -> residual."""
        x = self.input_proj(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class KVCompressedAttentionPartial(nn.Module):
    """PSA-style partial KV-compressed attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
    ):
        """Initialize partial KVCA."""
        super().__init__()
        if c2 % 2 != 0:
            raise ValueError(f"KVCompressedAttentionPartial requires even c2, got {c2}")
        c_attn = c2 // 2
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.attn = KVCompressedAttention(
            c_attn,
            c_attn,
            num_heads=max(1, num_heads // 2),
            sr_ratio=sr_ratio,
            mode=mode,
            attn_drop=attn_drop,
            residual=True,
        )
        self.out_proj = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KVCA on first half of channels, bypass second half, and mix outputs."""
        x = self.input_proj(x)
        x_attn, x_bypass = x.chunk(2, dim=1)
        x_attn = self.attn(x_attn)
        return self.out_proj(torch.cat([x_attn, x_bypass], dim=1))


class _TopKGroupKVAttentionBase(nn.Module):
    """Shared utilities for top-k grouped K/V attention blocks."""

    def __init__(self, c1: int, c2: int, num_heads: int, group_size: int):
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.group_size = max(1, int(group_size))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False)
        self.score = nn.Conv2d(c2, 1, 3, padding=1, bias=True)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to a multiple and return the original H/W."""
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def _to_groups(self, x: torch.Tensor, group_h: int, group_w: int) -> torch.Tensor:
        """Convert BCHW into B x groups x group_tokens x C."""
        b, c, _, _ = x.shape
        g = self.group_size
        return (
            x.view(b, c, group_h, g, group_w, g)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, group_h * group_w, g * g, c)
        )

    @staticmethod
    def _to_regions(x: torch.Tensor, region_size: int, region_h: int, region_w: int) -> torch.Tensor:
        """Convert BCHW into B x regions x region_tokens x C."""
        b, c, _, _ = x.shape
        r = region_size
        return (
            x.view(b, c, region_h, r, region_w, r)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, region_h * region_w, r * r, c)
        )

    def _compress_kv_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress projected K/V maps into one weighted token per spatial group."""
        padded, _, _ = self._pad_to_multiple(x, self.group_size)
        _, _, hp, wp = padded.shape
        group_h, group_w = hp // self.group_size, wp // self.group_size

        k_map, v_map = self.kv(padded).chunk(2, dim=1)
        score_map = self.score(padded)
        k_tokens = self._to_groups(k_map, group_h, group_w)
        v_tokens = self._to_groups(v_map, group_h, group_w)
        score_tokens = self._to_groups(score_map, group_h, group_w).squeeze(-1)
        weights = score_tokens.softmax(dim=2).unsqueeze(-1)

        k_groups = (k_tokens * weights).sum(dim=2)
        v_groups = (v_tokens * weights).sum(dim=2)
        group_scores = score_tokens.mean(dim=2)
        return k_groups, v_groups, group_scores

    def _format_full_q(self, q_map: torch.Tensor) -> torch.Tensor:
        """Convert full-resolution Q map to B x heads x tokens x head_dim."""
        b, c, h, w = q_map.shape
        q = q_map.flatten(2).transpose(1, 2)
        return q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _format_selected_kv(self, selected: torch.Tensor) -> torch.Tensor:
        """Convert B x tokens x C selected K/V tokens to B x heads x tokens x head_dim."""
        b, n, _ = selected.shape
        return selected.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


class TopKGlobalGroupKVAttention(_TopKGroupKVAttentionBase):
    """Full-query attention over a global top-k set of compressed K/V groups."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, group_size: int = 4, topk: int = 100):
        """Initialize global top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select one top-k group set per image and attend all query tokens to it."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        q = self._format_full_q(self.q(x))
        k_groups, v_groups, group_scores = self._compress_kv_groups(x)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = group_scores.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]
        k = self._format_selected_kv(k_groups[batch_idx, route_idx])
        v = self._format_selected_kv(v_groups[batch_idx, route_idx])

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_bn(self.proj(out))


class TopKAdaptiveGroupKVAttention(_TopKGroupKVAttentionBase):
    """Region-wise query attention over adaptive top-k compressed K/V groups."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        group_size: int = 4,
        query_region_size: int = 10,
        topk: int = 8,
    ):
        """Initialize adaptive top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.query_region_size = max(1, int(query_region_size))
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select top-k K/V groups per query region, then attend region tokens to them."""
        x = self.input_proj(x)
        b, c, _, _ = x.shape
        q_map = self.q(x)
        q_padded, orig_h, orig_w = self._pad_to_multiple(q_map, self.query_region_size)
        _, _, hp, wp = q_padded.shape
        region_h, region_w = hp // self.query_region_size, wp // self.query_region_size
        num_regions = region_h * region_w
        tokens_per_region = self.query_region_size * self.query_region_size

        q_regions = self._to_regions(q_padded, self.query_region_size, region_h, region_w)
        q_repr = q_regions.mean(dim=2)
        k_groups, v_groups, _ = self._compress_kv_groups(x)
        affinity = (q_repr @ k_groups.transpose(-2, -1)) * (c**-0.5)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]

        outputs = []
        for region_idx in range(num_regions):
            selected = route_idx[:, region_idx]
            q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
            q_tokens = q_tokens.permute(0, 2, 1, 3)
            k_tokens = self._format_selected_kv(k_groups[batch_idx, selected])
            v_tokens = self._format_selected_kv(v_groups[batch_idx, selected])
            with torch.cuda.amp.autocast(enabled=False):
                attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
            out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
            outputs.append(out)

        out_regions = torch.stack(outputs, dim=1)
        r = self.query_region_size
        out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
        return x + self.proj_bn(self.proj(out))


class BiLevelRoutingAttention(nn.Module):
    """NCHW bi-level routing attention adapted for YOLO feature maps."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        n_win: int = 7,
        topk: int = 4,
        side_dwconv: int = 3,
    ):
        """Initialize bi-level routing attention."""
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = c2**-0.5
        self.n_win = max(1, int(n_win))
        self.topk = max(1, int(topk))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.qkv_linear = nn.Conv2d(c2, c2 * 3, 1)
        self.lepe = (
            nn.Conv2d(c2, c2, side_dwconv, stride=1, padding=side_dwconv // 2, groups=c2)
            if side_dwconv > 0
            else None
        )
        self.output_linear = nn.Conv2d(c2, c2, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_region_size(x: torch.Tensor, region_size: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to multiples of region_size and return original H/W."""
        h, w = x.shape[-2:]
        pad_h = (region_size[0] - h % region_size[0]) % region_size[0]
        pad_w = (region_size[1] - w % region_size[1]) % region_size[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    @staticmethod
    def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
        """Convert BCHW to B x heads x regions x region_tokens x head_dim."""
        b, c, h, w = x.shape
        rh, rw = region_size
        region_h, region_w = h // rh, w // rw
        x = x.view(b, num_heads, c // num_heads, region_h, rh, region_w, rw)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        return x.view(b, num_heads, region_h * region_w, rh * rw, c // num_heads), region_h, region_w

    @staticmethod
    def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
        """Convert B x heads x regions x region_tokens x head_dim to BCHW."""
        b, num_heads, _, _, head_dim = x.shape
        rh, rw = region_size
        x = x.view(b, num_heads, region_h, region_w, rh, rw, head_dim)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        return x.view(b, num_heads * head_dim, region_h * rh, region_w * rw)

    def _regional_routing_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        region_graph: torch.Tensor,
        region_size: tuple[int, int],
    ) -> torch.Tensor:
        """Apply token attention from each query region to selected key/value regions."""
        query, orig_h, orig_w = self._pad_to_region_size(query, region_size)
        key, _, _ = self._pad_to_region_size(key, region_size)
        value, _, _ = self._pad_to_region_size(value, region_size)

        query, q_region_h, q_region_w = self._grid2seq(query, region_size, self.num_heads)
        key, _, _ = self._grid2seq(key, region_size, self.num_heads)
        value, _, _ = self._grid2seq(value, region_size, self.num_heads)

        b, num_heads, q_regions, topk = region_graph.shape
        _, _, kv_regions, kv_region_tokens, head_dim = key.shape
        index = region_graph.view(b, num_heads, q_regions, topk, 1, 1).expand(
            -1, -1, -1, -1, kv_region_tokens, head_dim
        )
        key_g = torch.gather(
            key.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )
        value_g = torch.gather(
            value.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )

        attn = (query * self.scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ value_g.flatten(-3, -2)
        out = self._seq2grid(out, q_region_h, q_region_w, region_size)
        return out[:, :, :orig_h, :orig_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route query regions to top-k key/value regions and apply token attention."""
        x = self.input_proj(x)
        _, c, h, w = x.shape
        region_size = (max(1, h // self.n_win), max(1, w // self.n_win))

        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=1)

        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)
        k_r = k_r.flatten(2, 3)
        affinity = q_r @ k_r
        route_count = min(self.topk, k_r.shape[-1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        route_idx = route_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        out = self._regional_routing_attention(q, k, v, route_idx, region_size)
        lepe = self.lepe(v) if self.lepe is not None else torch.zeros_like(v)
        out = out + lepe
        return x + self.output_bn(self.output_linear(out.reshape(-1, c, h, w)))


class RegionRoutingAttentionLite(nn.Module):
    """Bi-level routing attention for YOLO feature maps using pure PyTorch ops."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, region_size: int = 10, topk: int = 4):
        """Initialize region-routed token attention."""
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.region_size = max(1, int(region_size))
        self.topk = max(1, int(topk))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    def _pad_to_regions(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad H/W so both are divisible by region_size."""
        h, w = x.shape[-2:]
        pad_h = (self.region_size - h % self.region_size) % self.region_size
        pad_w = (self.region_size - w % self.region_size) % self.region_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def _to_regions(self, x: torch.Tensor, region_h: int, region_w: int) -> torch.Tensor:
        """Convert BCHW into B x regions x tokens x C."""
        b, c, _, _ = x.shape
        r = self.region_size
        return (
            x.view(b, c, region_h, r, region_w, r)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, region_h * region_w, r * r, c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each query region to top-k key/value regions, then attend locally."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        padded, orig_h, orig_w = self._pad_to_regions(x)
        _, _, hp, wp = padded.shape
        region_h, region_w = hp // self.region_size, wp // self.region_size
        num_regions = region_h * region_w
        tokens_per_region = self.region_size * self.region_size

        qkv = self.qkv(padded)
        q, k, v = qkv.chunk(3, dim=1)
        q_regions = self._to_regions(q, region_h, region_w)
        k_regions = self._to_regions(k, region_h, region_w)
        v_regions = self._to_regions(v, region_h, region_w)

        q_repr = q_regions.mean(dim=2)
        k_repr = k_regions.mean(dim=2)
        affinity = (q_repr @ k_repr.transpose(-2, -1)) * (c**-0.5)
        route_count = min(self.topk, num_regions)
        route_idx = affinity.topk(route_count, dim=-1).indices

        outputs = []
        batch_idx = torch.arange(b, device=x.device)[:, None]
        for region_idx in range(num_regions):
            selected = route_idx[:, region_idx]
            q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
            q_tokens = q_tokens.permute(0, 2, 1, 3)
            k_tokens = k_regions[batch_idx, selected].reshape(
                b, route_count * tokens_per_region, self.num_heads, self.head_dim
            )
            v_tokens = v_regions[batch_idx, selected].reshape(
                b, route_count * tokens_per_region, self.num_heads, self.head_dim
            )
            k_tokens = k_tokens.permute(0, 2, 1, 3)
            v_tokens = v_tokens.permute(0, 2, 1, 3)
            attn = (q_tokens @ k_tokens.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
            outputs.append(out)

        out_regions = torch.stack(outputs, dim=1)
        r = self.region_size
        out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
        return x + self.proj_bn(self.proj(out))


class P4CrossScaleLocalAttention(nn.Module):
    """P4-centered local cross-scale fusion for small-object detection heads.

    The block receives [P2, P3, P4, P5], keeps P4 as the output scale, attends P4
    queries to local windows from P3/P4/P5, then uses P2 as a localization gate.
    It is intended as a 3-head ablation: Detect still consumes [P3, P4*, P5].
    """

    def __init__(
        self,
        channels: list[int],
        c2: int,
        num_heads: int = 4,
        window_size: int = 7,
        p2_gate_scale: float = 0.1,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"P4CrossScaleLocalAttention expects [P2, P3, P4, P5] channels, got {channels}")

        c_p2, c_p3, c_p4, c_p5 = channels
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = max(1, int(window_size))
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.p2_gate_scale = float(p2_gate_scale)

        self.p2_proj = Conv(c_p2, c2, 1, 1)
        self.p3_proj = Conv(c_p3, c2, 1, 1)
        self.p4_proj = Conv(c_p4, c2, 1, 1)
        self.p5_proj = Conv(c_p5, c2, 1, 1)

        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.k = nn.Conv2d(c2, c2, 1, bias=False)
        self.v = nn.Conv2d(c2, c2, 1, bias=False)
        self.unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size // 2)
        self.out = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2))
        self.gate = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _local_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        tokens = self.window_size * self.window_size
        k = self.unfold(self.k(x)).view(b, self.num_heads, self.head_dim, tokens, h * w)
        v = self.unfold(self.v(x)).view(b, self.num_heads, self.head_dim, tokens, h * w)
        k = k.permute(0, 1, 4, 3, 2).contiguous()
        v = v.permute(0, 1, 4, 3, 2).contiguous()
        return k, v

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse [P2, P3, P4, P5] into an enhanced P4 tensor."""
        p2, p3, p4, p5 = x
        h, w = p4.shape[-2:]

        p4 = self.p4_proj(p4)
        p3 = self._resize_to(self.p3_proj(p3), (h, w))
        p5 = self._resize_to(self.p5_proj(p5), (h, w))
        p2_gate = self.gate(self._resize_to(self.p2_proj(p2), (h, w)))

        b, c, _, _ = p4.shape
        q = self.q(p4).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        keys, values = [], []
        for feature in (p3, p4, p5):
            k, v = self._local_tokens(feature)
            keys.append(k)
            values.append(v)
        k = torch.cat(keys, dim=3)
        v = torch.cat(values, dim=3)

        attn = (q.unsqueeze(3) * k).sum(dim=-1) * self.scale
        attn = attn.softmax(dim=-1)
        refined = (attn.unsqueeze(-1) * v).sum(dim=3)
        refined = refined.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        refined = self.out(refined) * (1.0 + self.p2_gate_scale * p2_gate)
        return p4 + self.gamma * refined


class LDown(nn.Module):
    """Lightweight downsample: depthwise spatial reduction followed by pointwise channel projection.

    This is a paper-inspired implementation of HierLight-YOLO's LDown idea, adapted to the local
    Ultralytics Conv/DWConv style instead of copying an upstream implementation.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, act: bool = True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample spatially, then project channels."""
        return self.pw(self.dw(x))


class IRDCBUnit(nn.Module):
    """Inverted residual depthwise unit used inside IRDCB."""

    def __init__(self, c: int, expansion: float = 2.0, shortcut: bool = True):
        super().__init__()
        hidden = max(8, int(c * expansion))
        self.expand = Conv(c, hidden, 1, 1)
        self.dw1 = Conv(hidden, hidden, 3, 1, g=hidden)
        self.dw2 = Conv(hidden, hidden, 3, 1, g=hidden)
        self.project = Conv(hidden, c, 1, 1, act=False)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expand, two depthwise spatial filters, and projection."""
        y = self.project(self.dw2(self.dw1(self.expand(x))))
        return x + y if self.shortcut else y


class IRDCB(nn.Module):
    """C2f-compatible inverted residual depthwise convolution block.

    Inspired by HierLight-YOLO's IRDCB, this implementation keeps the YOLOv8 C2f-style split/concat
    interface so it can replace selected neck C2f layers without changing surrounding YAML topology.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, expansion: float = 2.0, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(IRDCBUnit(self.c, expansion=expansion, shortcut=shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using C2f-style feature accumulation with IRDCB units."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SAGRI(nn.Module):
    """Fuse high-resolution P2 detail into P3 without exposing P2 to Detect."""

    def __init__(
        self,
        channels: list[int],
        c2: int,
        k: int = 3,
        gamma_init: float = 0.0,
        gate_scale: float = 1.0,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"SAGRI expects [P2, P3] channels, got {channels}")
        c_p2, c_p3 = channels
        self.p2_proj = Conv(c_p2, c2, 1, 1)
        self.p3_proj = Conv(c_p3, c2, 1, 1)
        self.gate = nn.Sequential(
            Conv(2 * c2, c2, k, 1, g=c2),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid(),
        )
        self.context_down = LDown(c2, c2, k=3, s=2)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gate_scale = float(gate_scale)

    @staticmethod
    def _resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _to_p3_size(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == target_size:
            return x
        if x.shape[-2] == target_size[0] * 2 and x.shape[-1] == target_size[1] * 2:
            return self.context_down(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return a P3-scale tensor enriched by local P2 detail."""
        p2, p3 = x
        target_size = p3.shape[-2:]
        p3_base = self.p3_proj(p3)
        p2_detail = self.p2_proj(p2)
        p3_up = self._resize_to(p3_base, p2_detail.shape[-2:])
        gate = self.gate(torch.cat((p2_detail, p3_up), dim=1))
        context = self._to_p3_size(gate * p2_detail, target_size)
        return p3_base + self.gamma * self.gate_scale * context


class SAGRIConvDown(SAGRI):
    """P2-to-P3 gated injection using standard stride-2 convolution instead of LDown."""

    def __init__(
        self,
        channels: list[int],
        c2: int,
        k: int = 3,
        gamma_init: float = 0.0,
        gate_scale: float = 1.0,
    ):
        super().__init__(channels, c2, k=k, gamma_init=gamma_init, gate_scale=gate_scale)
        self.context_down = Conv(c2, c2, k=3, s=2)


class P5LocalToP4(nn.Module):
    """Fuse low-resolution P5 semantic context into P4 without exposing P5 to Detect."""

    def __init__(
        self,
        channels: list[int],
        c2: int,
        k: int = 3,
        gamma_init: float = 0.0,
        gate_scale: float = 1.0,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"P5LocalToP4 expects [P5, P4] channels, got {channels}")
        c_p5, c_p4 = channels
        self.p5_proj = Conv(c_p5, c2, 1, 1)
        self.p4_proj = Conv(c_p4, c2, 1, 1)
        self.gate = nn.Sequential(
            Conv(2 * c2, c2, k, 1, g=c2),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gate_scale = float(gate_scale)

    @staticmethod
    def _resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return a P4-scale tensor enriched by gated P5 semantic context."""
        p5, p4 = x
        p4_base = self.p4_proj(p4)
        p5_semantic = self._resize_to(self.p5_proj(p5), p4_base.shape[-2:])
        gate = self.gate(torch.cat((p5_semantic, p4_base), dim=1))
        return p4_base + self.gamma * self.gate_scale * gate * p5_semantic


class P2PartialKVToP3(nn.Module):
    """Partial cross-scale KV attention from compressed P2 keys/values into P3 queries."""

    def __init__(
        self,
        channels: list[int],
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 4,
        attn_ratio: float = 0.25,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"P2PartialKVToP3 expects [P2, P3] channels, got {channels}")
        c_p2, c_p3 = channels
        self.p2_proj = Conv(c_p2, c2, 1, 1)
        self.p3_proj = Conv(c_p3, c2, 1, 1)
        self.attn_c = max(8, int(c2 * attn_ratio))
        self.attn_c = min(self.attn_c, c2)
        self.num_heads = _choose_attention_heads(self.attn_c, num_heads)
        self.head_dim = self.attn_c // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = max(1, int(sr_ratio))
        self.q = nn.Conv2d(self.attn_c, self.attn_c, 1, bias=False)
        self.k = nn.Conv2d(self.attn_c, self.attn_c, 1, bias=False)
        self.v = nn.Conv2d(self.attn_c, self.attn_c, 1, bias=False)
        self.out = nn.Sequential(nn.Conv2d(self.attn_c, c2, 1, bias=False), nn.BatchNorm2d(c2))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _compress_p2(self, x: torch.Tensor) -> torch.Tensor:
        if self.sr_ratio <= 1:
            return x
        if x.shape[-2] < self.sr_ratio or x.shape[-1] < self.sr_ratio:
            return F.adaptive_avg_pool2d(x, (1, 1))
        return F.avg_pool2d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Apply partial cross-attention and return a P3-scale tensor."""
        p2, p3 = x
        p3_base = self.p3_proj(p3)
        p2_base = self.p2_proj(p2)
        q_src = p3_base[:, : self.attn_c]
        kv_src = self._compress_p2(p2_base[:, : self.attn_c])

        b, _, h, w = q_src.shape
        q = self.q(q_src).flatten(2).transpose(1, 2).reshape(b, h * w, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = self.k(kv_src).flatten(2).transpose(1, 2).reshape(b, -1, self.num_heads, self.head_dim)
        v = self.v(kv_src).flatten(2).transpose(1, 2).reshape(b, -1, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        refined = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, h * w, self.attn_c)
        refined = refined.transpose(1, 2).reshape(b, self.attn_c, h, w)
        return p3_base + self.gamma * self.out(refined)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fKV(nn.Module):
    """C2f block with PSA-style split and gated KV-compressed attention on the final hidden feature."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
    ):
        """Initialize a C2f-style block with a lightweight PSA-style KV attention branch."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.kv_c = max(1, self.c // 2)
        self.bypass_c = self.c - self.kv_c
        self.kv = KVCompressedAttention(self.kv_c, self.kv_c, num_heads, sr_ratio, mode)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
        """Refine one split of the final hidden feature with gated KV attention."""
        if self.bypass_c == 0:
            return x + self.gamma * (self.kv(x) - x)
        a, b = x.split((self.bypass_c, self.kv_c), dim=1)
        b = b + self.gamma * (self.kv(b) - b)
        return torch.cat((a, b), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f with PSA-style KV refinement on the final hidden feature."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            semantic = self.semseg(feat)
            return (p, semantic)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
