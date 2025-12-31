import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict
from SGF import SGFMemoryEfficient

# ----------------------------
# Utils
# ----------------------------
def _minmax_norm(x: torch.Tensor, eps: float = 1e-6, denom_floor: float = 1e-3) -> torch.Tensor:
    n = x.shape[0]
    flat = x.view(n, -1)
    x_min = flat.min(dim=1, keepdim=True).values.view(n, 1, 1, 1)
    x_max = flat.max(dim=1, keepdim=True).values.view(n, 1, 1, 1)
    denom = (x_max - x_min).clamp_min(denom_floor)  # 关键：避免分母过小
    return (x - x_min) / (denom + eps)


def _ensure_nchw_rgb(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("Ia/Ib must be 4D tensors (N,C,H,W).")
    if a.size(1) != 3 or b.size(1) != 3:
        raise ValueError("Ia/Ib must be RGB (C=3).")
    if a.shape[-2:] != b.shape[-2:]:
        raise ValueError("Ia and Ib must have the same H,W.")


# ----------------------------
# Structure Extractor Interface
# ----------------------------
class StructureExtractorBase(nn.Module):
    """
    Returns single-channel structure feature map FD: (N,1,H,W).
    Input default: concatenated image I = cat(Ia, Ib) -> (N,6,H,W).
    """
    def forward(self, I: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ----------------------------
# 1) Torch Sobel / Laplacian Extractors (operators)
#    Correspond to FD = D(I) for traditional operators. :contentReference[oaicite:7]{index=7}
# ----------------------------
class SobelExtractor(StructureExtractorBase):
    """
    Computes Sobel gradient magnitude on luminance of concatenated input.
    """
    def __init__(self, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.eps = eps
        self.normalize = normalize

        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if I.dim() != 4 or I.size(1) != 6:
            raise ValueError("SobelExtractor expects I of shape (N,6,H,W).")

        # Convert 6ch to 1ch by averaging two RGB luminances
        Ia, Ib = I[:, :3], I[:, 3:]
        # simple luminance
        Ya = 0.2989 * Ia[:, 0:1] + 0.5870 * Ia[:, 1:2] + 0.1140 * Ia[:, 2:3]
        Yb = 0.2989 * Ib[:, 0:1] + 0.5870 * Ib[:, 1:2] + 0.1140 * Ib[:, 2:3]
        Y = 0.5 * (Ya + Yb)  # (N,1,H,W)

        Yp = F.pad(Y, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(Yp, self.kx)
        gy = F.conv2d(Yp, self.ky)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)

        return _minmax_norm(mag, self.eps) if self.normalize else mag


class LaplacianExtractor(StructureExtractorBase):
    """
    Computes Laplacian magnitude on luminance of concatenated input.
    """
    def __init__(self, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.eps = eps
        self.normalize = normalize

        k = torch.tensor([[0,  1, 0],
                          [1, -4, 1],
                          [0,  1, 0]], dtype=torch.float32)
        self.register_buffer("k", k.view(1, 1, 3, 3))

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if I.dim() != 4 or I.size(1) != 6:
            raise ValueError("LaplacianExtractor expects I of shape (N,6,H,W).")

        Ia, Ib = I[:, :3], I[:, 3:]
        Ya = 0.2989 * Ia[:, 0:1] + 0.5870 * Ia[:, 1:2] + 0.1140 * Ia[:, 2:3]
        Yb = 0.2989 * Ib[:, 0:1] + 0.5870 * Ib[:, 1:2] + 0.1140 * Ib[:, 2:3]
        Y = 0.5 * (Ya + Yb)

        Yp = F.pad(Y, (1, 1, 1, 1), mode="reflect")
        lap = F.conv2d(Yp, self.k)
        mag = lap.abs()

        return _minmax_norm(mag, self.eps) if self.normalize else mag


# ----------------------------
# 2) Deep extractor wrapper (RCF/BDCN/EDTER etc.)
#    Uses mid-level features + channel-wise averaging, Eq.(10). :contentReference[oaicite:8]{index=8}
# ----------------------------
class DeepMidFeatureExtractor(StructureExtractorBase):
    """
    Wrap an arbitrary deep edge model and extract a specified intermediate feature map via hook,
    then channel-average to produce single-channel FD.

    You provide:
      - model: nn.Module
      - hook_layer: module name (string) to hook (e.g. "backbone.layer2.1.conv2")
    """
    def __init__(
        self,
        model: nn.Module,
        hook_layer: str,
        freeze: bool = True,
        eps: float = 1e-6,
        normalize: bool = True,
    ):
        super().__init__()
        self.model = model
        self.hook_layer = hook_layer
        self.eps = eps
        self.normalize = normalize

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self._feat: Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self) -> None:
        modules: Dict[str, nn.Module] = dict(self.model.named_modules())
        if self.hook_layer not in modules:
            raise ValueError(f"hook_layer '{self.hook_layer}' not found in model. "
                             f"Available keys example: {list(modules.keys())[:20]} ...")

        def _hook(_m, _inp, out):
            # Expect out: (N,C,H,W)
            self._feat = out

        modules[self.hook_layer].register_forward_hook(_hook)

    @torch.no_grad()
    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if I.dim() != 4 or I.size(1) != 6:
            raise ValueError("DeepMidFeatureExtractor expects I of shape (N,6,H,W).")

        self._feat = None
        _ = self.model(I)  # forward to trigger hook

        if self._feat is None:
            raise RuntimeError("Hook did not capture features. Check hook_layer and model forward.")

        Q = self._feat
        if Q.dim() != 4:
            raise RuntimeError("Captured feature is not 4D (N,C,H,W).")

        # Eq.(10): channel-wise averaging -> single channel
        FD = Q.mean(dim=1, keepdim=True)

        # Upsample to input size if needed
        if FD.shape[-2:] != I.shape[-2:]:
            FD = F.interpolate(FD, size=I.shape[-2:], mode="bilinear", align_corners=False)

        return _minmax_norm(FD, self.eps) if self.normalize else FD


def build_structure_extractor(name: str, **kwargs) -> StructureExtractorBase:
    """
    Factory for flexible selection. Paper evaluates Sobel/Laplace and deep models etc. :contentReference[oaicite:9]{index=9}
    """
    name = name.lower()
    if name == "sobel":
        return SobelExtractor(**kwargs)
    if name == "laplace" or name == "laplacian":
        return LaplacianExtractor(**kwargs)
    if name == "deep_mid":
        return DeepMidFeatureExtractor(**kwargs)
    raise ValueError(f"Unknown extractor: {name}")

# -----------------------------
# Basic blocks
# -----------------------------
def conv3x3(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Conv2d:
    """3x3 conv with padding that keeps spatial size."""
    return nn.Conv2d(
        in_ch, out_ch,
        kernel_size=3,
        stride=1,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        if k == 3:
            self.conv = conv3x3(in_ch, out_ch, dilation=dilation)
        elif k == 1:
            self.conv = conv1x1(in_ch, out_ch)
        else:
            raise ValueError("Only k=1 or k=3 supported.")
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LayerNorm2d(nn.Module):
    """
    True LayerNorm over channel dimension for each spatial location:
    x: (N, C, H, W)
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N,C,H,W) -> (N,H,W,C) -> LN over C -> back
        n, c, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2).contiguous()


# -----------------------------
# CAM (Context-Aware Module)
# -----------------------------
class CAM(nn.Module):
    """
    A context modeling + transform + residual add block consistent with Fig.6 notion:
      - Context modeling (spatial softmax weighting)
      - 1x1 conv -> LayerNorm -> ReLU -> 1x1 conv
      - Add residual to input
    """
    def __init__(self, channels: int, reduction: int = 1):
        """
        reduction: optional bottleneck ratio. To strictly follow "Conv1x1 ... Conv1x1",
                   you can keep reduction=1 (default, no bottleneck).
                   If you want a bottleneck, set reduction=16, etc.
        """
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        mid = max(channels // reduction, 1)

        # Produce attention logits (N,1,H,W) then softmax over H*W
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        # Transform on context vector (N,C,1,1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            LayerNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W)
        """
        n, c, h, w = x.shape

        # Spatial attention
        attn_logits = self.attn_conv(x)               # (N,1,H,W)
        attn = attn_logits.view(n, 1, h * w)          # (N,1,HW)
        attn = F.softmax(attn, dim=2)                 # softmax over spatial

        # Weighted global context: sum over spatial
        x_flat = x.view(n, c, h * w)                  # (N,C,HW)
        context = torch.sum(x_flat * attn, dim=2, keepdim=True)  # (N,C,1)
        context = context.view(n, c, 1, 1)            # (N,C,1,1)

        # Transform and residual add
        delta = self.transform(context)               # (N,C,1,1)
        return x + delta


# -----------------------------
# MARM (Multi-scale Adaptive Residual Module)
# -----------------------------
class MARM(nn.Module):
    """
    Multi-scale residual module with parallel dilated convs (D=1,2,3),
    concatenation, channel compression, embedded CAM, and residual add.
    """
    def __init__(self, channels: int = 64, cam_reduction: int = 1):
        super().__init__()
        self.branch_d1 = ConvBNReLU(channels, channels, k=3, dilation=1)
        self.branch_d2 = ConvBNReLU(channels, channels, k=3, dilation=2)
        self.branch_d3 = ConvBNReLU(channels, channels, k=3, dilation=3)

        # After concat: 3C -> C
        self.fuse = ConvBNReLU(channels * 3, channels, k=1)

        # Embedded CAM
        self.cam = CAM(channels, reduction=cam_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b3 = self.branch_d3(x)

        y = torch.cat([b1, b2, b3], dim=1)  # (N,3C,H,W)
        y = self.fuse(y)                   # (N,C,H,W)
        y = self.cam(y)                    # (N,C,H,W)

        return y + residual


# -----------------------------
# Focus Detection Branch (N=6 MARMs)
# -----------------------------
class FocusDetectionBranch(nn.Module):
    """
    Focus detection branch:
      Input: concat(Ia, Ib) -> 6ch
      Stem: 3x3 conv(6->64)+BN+ReLU
      Backbone: N=6 MARMs (each with D=1,2,3 + embedded CAM + residual)
      Tail: independent CAM
      Output local map X: mean over channels -> normalize to [0,1]
    """
    def __init__(
        self,
        in_channels: int = 6,
        channels: int = 64,
        num_marm: int = 6,          # user requested N=6
        cam_reduction: int = 1,     # set to 1 to match "Conv1x1 ... Conv1x1" without bottleneck
        norm_eps: float = 1e-6,
        x_normalize: str = "minmax" # "minmax" or "sigmoid"
    ):
        super().__init__()
        if x_normalize not in ("minmax", "sigmoid"):
            raise ValueError("x_normalize must be 'minmax' or 'sigmoid'")

        self.channels = channels
        self.num_marm = num_marm
        self.norm_eps = norm_eps
        self.x_normalize = x_normalize

        self.stem = ConvBNReLU(in_channels, channels, k=3, dilation=1)

        self.marms = nn.ModuleList([
            MARM(channels=channels, cam_reduction=cam_reduction)
            for _ in range(num_marm)
        ])

        # Independent CAM after the last MARM
        self.tail_cam = CAM(channels, reduction=cam_reduction)

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming init for convs; BN to default 1/0.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,1,H,W)
        """
        if self.x_normalize == "sigmoid":
            return torch.sigmoid(x)

        # Per-sample min-max normalization to [0,1]
        n = x.shape[0]
        x_flat = x.view(n, -1)
        x_min = x_flat.min(dim=1, keepdim=True).values.view(n, 1, 1, 1)
        x_max = x_flat.max(dim=1, keepdim=True).values.view(n, 1, 1, 1)
        #return (x - x_min) / (x_max - x_min + self.norm_eps)
        denom = (x_max - x_min).clamp_min(1e-3)
        return (x - x_min) / (denom + self.norm_eps)

    def forward(self, ia: torch.Tensor, ib: torch.Tensor = None):
        """
        Supports two calling styles:
          1) forward(ia, ib): ia/ib are (N,3,H,W) -> concat -> (N,6,H,W)
          2) forward(x): ia already is (N,6,H,W), ib=None

        Returns:
          feat: (N,64,H,W)  - final feature after tail CAM
          x_local: (N,1,H,W) - single-channel local map X for SGF
        """
        if ib is None:
            x = ia
            if x.dim() != 4 or x.size(1) != 6:
                raise ValueError("If ib is None, ia must be a 6-channel tensor (N,6,H,W).")
        else:
            if ia.dim() != 4 or ib.dim() != 4:
                raise ValueError("ia and ib must be 4D tensors (N,C,H,W).")
            if ia.size(1) != 3 or ib.size(1) != 3:
                raise ValueError("ia and ib must be RGB tensors with 3 channels each.")
            if ia.shape[-2:] != ib.shape[-2:]:
                raise ValueError("ia and ib must have the same H,W.")
            x = torch.cat([ia, ib], dim=1)  # (N,6,H,W)

        feat = self.stem(x)                 # (N,64,H,W)
        for block in self.marms:
            feat = block(feat)             # (N,64,H,W)
        feat = self.tail_cam(feat)          # (N,64,H,W)

        # Mean averaging -> (N,1,H,W), then normalization
        x_local = feat.mean(dim=1, keepdim=True)
        x_local = self._normalize_x(x_local)

        return feat, x_local


# ----------------------------
# Transformer wrapper (ResT is recommended in paper) :contentReference[oaicite:10]{index=10}
# We keep it pluggable so you can swap in your ResT implementation.
# ----------------------------
class StructureAwareBranch(nn.Module):
    """
    Pipeline:
      Ia/Ib -> concat I (N,6,H,W)
      FD = extractor(I)  -> (N,1,H,W)
      T  = transformer(FD)  -> (N,Ct,Ht,Wt) or (N,1,Ht,Wt)
      G  = to_guidance(T) -> upsample -> (N,1,H,W), normalized [0,1]
    """
    def __init__(
        self,
        extractor: StructureExtractorBase,
        transformer: nn.Module,
        transformer_in_channels: int = 1,
        guidance_from: str = "mean",   # "mean" or "conv1x1"
        eps: float = 1e-6,
    ):
        super().__init__()
        if guidance_from not in ("mean", "conv1x1"):
            raise ValueError("guidance_from must be 'mean' or 'conv1x1'.")

        self.extractor = extractor
        self.transformer = transformer
        self.eps = eps
        self.guidance_from = guidance_from

        # Optional adapter if transformer expects >1 channel input
        self.in_adapter = None
        if transformer_in_channels != 1:
            self.in_adapter = nn.Conv2d(1, transformer_in_channels, kernel_size=1, bias=False)

        # If using conv1x1 to produce G from transformer feature
        self.to_g = None
        if guidance_from == "conv1x1":
            # We'll lazily create it on first forward if Ct is unknown.
            self.to_g = None

    def _make_to_g(self, ct: int) -> None:
        self.to_g = nn.Conv2d(ct, 1, kernel_size=1, bias=True)

    def forward(self, ia: torch.Tensor, ib: torch.Tensor):
        _ensure_nchw_rgb(ia, ib)
        I = torch.cat([ia, ib], dim=1)  # (N,6,H,W)

        FD = self.extractor(I)          # (N,1,H,W)
        if FD.dim() != 4 or FD.size(1) != 1:
            raise RuntimeError("Extractor must output FD of shape (N,1,H,W).")

        x = FD
        if self.in_adapter is not None:
            x = self.in_adapter(x)

        T = self.transformer(x)  # expected (N,Ct,Ht,Wt) or (N,1,Ht,Wt)
        if T.dim() != 4:
            raise RuntimeError("Transformer must output 4D feature map (N,C,H,W).")

        # Build G
        if self.guidance_from == "mean":
            G = T.mean(dim=1, keepdim=True)   # (N,1,Ht,Wt)
        else:
            if self.to_g is None:
                self._make_to_g(T.size(1))
            G = self.to_g(T)                  # (N,1,Ht,Wt)

        # Upsample to original size for SGF: G ∈ R^{H×W} :contentReference[oaicite:11]{index=11}
        if G.shape[-2:] != ia.shape[-2:]:
            G = F.interpolate(G, size=ia.shape[-2:], mode="bilinear", align_corners=False)

        # Normalize for stability (recommended for SGF weights computation)
        G = _minmax_norm(G, self.eps)
        return FD, T, G


# ============================================================
# ResT (Efficient Transformer) implementation (self-contained)
# - Overlap Patch Embedding (conv)
# - Efficient MHSA with spatial reduction (sr_ratio)
# - MLP with DWConv
# - Conv Positional Encoding (depthwise conv)
# - Returns 4D feature map (N,C,H,W) for StructureAwareBranch
# ============================================================

class DropPath(nn.Module):
    """Stochastic Depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (rand < keep_prob).to(x.dtype)
        return x / keep_prob * mask


class LayerNormTokens(nn.Module):
    """LayerNorm over last dim for token tensor (N, L, C)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class DWConv(nn.Module):
    """Depthwise conv used inside MLP (token -> 2D -> token)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (N, L, C) -> (N, C, H, W) -> dwconv -> back
        N, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(N, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (N, L, C)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = DWConv(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """
    Efficient MHSA with spatial-reduction on K/V (sr_ratio).
    Token input: (N, L, C) where L=H*W
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(sr_ratio)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if self.sr_ratio > 1:
            # spatial reduction conv on 2D map
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, padding=0, groups=dim, bias=True)
            self.norm = LayerNormTokens(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        N, L, C = x.shape

        # Q
        q = self.q(x).reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, heads, L, d)

        # K,V (optionally reduced)
        if self.sr_ratio > 1:
            x_2d = x.transpose(1, 2).contiguous().view(N, C, H, W)  # (N,C,H,W)
            x_2d = self.sr(x_2d)                                     # (N,C,H',W')
            Hr, Wr = x_2d.shape[-2], x_2d.shape[-1]
            x_reduced = x_2d.flatten(2).transpose(1, 2).contiguous()  # (N, Lr, C)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced)                                  # (N, Lr, 2C)
        else:
            kv = self.kv(x)                                          # (N, L, 2C)

        kv = kv.reshape(N, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (N, heads, Lk, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale               # (N, heads, L, Lk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                              # (N, heads, L, d)
        out = out.transpose(1, 2).contiguous().reshape(N, L, C)     # (N, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ConvPosEnc(nn.Module):
    """
    Convolutional positional encoding (depthwise conv on 2D, then add).
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        N, L, C = x.shape
        x_2d = x.transpose(1, 2).contiguous().view(N, C, H, W)
        x_2d = x_2d + self.proj(x_2d)
        x = x_2d.flatten(2).transpose(1, 2).contiguous()
        return x


class ResTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.cpe = ConvPosEnc(dim, kernel_size=3)

        self.norm1 = LayerNormTokens(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)

        self.norm2 = LayerNormTokens(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Conv positional encoding
        x = self.cpe(x, H, W)

        # MHSA
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding with Conv2d.
    Returns tokens (N, L, C) and spatial size (H, W).
    """
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = LayerNormTokens(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (N,C,H,W)
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2).contiguous()  # (N, L, C)
        x = self.norm(x)
        return x, H, W


class ResT(nn.Module):
    """
    A practical ResT-Small style backbone (sufficient for SD-Fuse structure branch):
    - 4 stages
    - returns feature map from a chosen stage as 4D (N,C,H,W)

    out_stage: 0/1/2/3 (default 2 -> stage3, better spatial resolution than stage4)
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dims=(64, 128, 256, 512),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        sr_ratios=(8, 4, 2, 1),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_stage: int = 2,
    ):
        super().__init__()
        assert 0 <= out_stage <= 3
        self.out_stage = out_stage

        # Patch Embeds (overlap)
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_chans,        embed_dims[0], kernel_size=7, stride=4, padding=3),
            OverlapPatchEmbed(embed_dims[0],   embed_dims[1], kernel_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[1],   embed_dims[2], kernel_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[2],   embed_dims[3], kernel_size=3, stride=2, padding=1),
        ])

        # Stochastic depth decay
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ResTBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        sr_ratio=sr_ratios[i],
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                    )
                )
            cur += depths[i]
            self.stages.append(nn.ModuleList(blocks))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_chans, H, W)
        return: feature map from out_stage as (N, C, Hs, Ws)
        """
        out = None
        for stage_idx in range(4):
            x, H, W = self.patch_embeds[stage_idx](x)  # tokens
            for blk in self.stages[stage_idx]:
                x = blk(x, H, W)
            # reshape tokens -> feature map
            N, L, C = x.shape
            feat = x.transpose(1, 2).contiguous().view(N, C, H, W)

            if stage_idx == self.out_stage:
                out = feat
                break
            # next stage takes 4D feature map as input
            x = feat
        assert out is not None
        return out


# -----------------------------
# Decision head AFTER SGF (Conv + Sigmoid)
# -----------------------------
class DecisionHead(nn.Module):
    """
    After SGF, the refined map is single-channel (N,1,H,W).
    Apply Conv (default 1x1) + Sigmoid to obtain decision map DM.
    """
    def __init__(self, k: int = 1):
        super().__init__()
        if k not in (1, 3):
            raise ValueError("k must be 1 or 3")
        if k == 1:
            self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4 or z.size(1) != 1:
            raise ValueError("DecisionHead expects input z of shape (N,1,H,W).")
        return torch.sigmoid(self.conv(z))


class SD_Fuse_Net(nn.Module):
    def __init__(
        self,
        extractor,               # StructureExtractorBase 实例
        transformer,             # ResT 实例
        r: int = 5,
        lam: float = 0.5,
        focus_channels: int = 64,
        num_marm: int = 6,
        guidance_from: str = "mean",
        transformer_in_channels: int = 1,
    ):
        super().__init__()

        # ---- Focus branch ----
        self.focus_branch = FocusDetectionBranch(
            in_channels=6,
            channels=focus_channels,
            num_marm=num_marm,
            cam_reduction=1,
            x_normalize="minmax",
        )

        # ---- Structure branch ----
        self.structure_branch = StructureAwareBranch(
            extractor=extractor,
            transformer=transformer,
            transformer_in_channels=transformer_in_channels,
            guidance_from=guidance_from,
        )

        # ---- SGF ----
        self.sgf = SGFMemoryEfficient(r=r, lam=lam)

        # ---- Decision head ----
        self.dm_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    def forward(self, Ia: torch.Tensor, Ib: torch.Tensor, return_logits: bool = False):
        # Focus -> X
        _, X = self.focus_branch(Ia, Ib)

        # Structure -> G
        _, _, G = self.structure_branch(Ia, Ib)

        X = X.clamp(0.0, 1.0)
        G = G.clamp(0.0, 1.0)

        # SGF -> Z
        Z = self.sgf(X, G)

        # logits
        logits = self.dm_conv(Z)  # (B,1,H,W)

        if return_logits:
            return logits
        return torch.sigmoid(logits)