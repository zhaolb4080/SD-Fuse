import torch
import torch.nn as nn
import torch.nn.functional as F


def _box_mean(x: torch.Tensor, r: int, pad_mode: str = "reflect") -> torch.Tensor:
    """
    Per-pixel box mean with radius r using avg_pool2d.
    Uses reflect/replicate padding to avoid border bias (recommended).
    x: [B,C,H,W]
    """
    if r <= 0:
        return x
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode=pad_mode)  # (left,right,top,bottom)
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)


def _grad_mag(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """||∇G|| via finite differences (differentiable). g: [B,1,H,W]"""
    gx = F.pad(g[..., :, 1:] - g[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(g[..., 1:, :] - g[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + eps)


def _local_var(x: torch.Tensor, r: int, eps: float = 1e-12, pad_mode: str = "reflect"):
    mu = _box_mean(x, r, pad_mode=pad_mode)
    mu2 = _box_mean(x * x, r, pad_mode=pad_mode)
    var = (mu2 - mu * mu).clamp_min(0.0)
    return mu, var + eps


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    r: int,
    c1: float = 1e-4,   # 0.01^2
    c2: float = 9e-4,   # 0.03^2
    eps: float = 1e-12,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Local SSIM over (2r+1)x(2r+1) window. Fully differentiable.
    x,y: [B,1,H,W] (recommended) or [B,C,H,W] (must match)
    """
    if x.shape != y.shape:
        raise ValueError(f"SSIM requires same shape. x={x.shape}, y={y.shape}")

    mux = _box_mean(x, r, pad_mode=pad_mode)
    muy = _box_mean(y, r, pad_mode=pad_mode)

    mu_x2 = mux * mux
    mu_y2 = muy * muy
    mu_xy = mux * muy

    sig_x2 = (_box_mean(x * x, r, pad_mode=pad_mode) - mu_x2).clamp_min(0.0)
    sig_y2 = (_box_mean(y * y, r, pad_mode=pad_mode) - mu_y2).clamp_min(0.0)
    sig_xy = _box_mean(x * y, r, pad_mode=pad_mode) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    return num / (den + eps)


class SGFMemoryEfficient(nn.Module):
    """
    Memory-efficient & fully differentiable SGF:
    - no unfold / no per-pixel loop
    - local stats via avg_pool2d
    - no torch.no_grad(), no detach()
    """
    def __init__(
        self,
        r: int = 5,
        lam: float = 1.0,
        eps: float = 1e-6,
        theta: float = 1e-3,
        use_softmin_eta: bool = False,
        softmin_beta: float = 20.0,
        pad_mode: str = "reflect",  # "reflect" or "replicate"
    ):
        super().__init__()
        self.r = int(r)
        self.lam = float(lam)
        self.eps = float(eps)
        self.theta = float(theta)
        self.use_softmin_eta = bool(use_softmin_eta)
        self.softmin_beta = float(softmin_beta)
        self.pad_mode = pad_mode

    def _eta(self, chi: torch.Tensor) -> torch.Tensor:
        mu = chi.mean(dim=(-2, -1), keepdim=True)
        if self.use_softmin_eta:
            b = self.softmin_beta
            mn = -(torch.logsumexp(-b * chi, dim=(-2, -1), keepdim=True) / b)
        else:
            mn = chi.amin(dim=(-2, -1), keepdim=True)
        return (mu - mn).clamp_min(self.eps)

    def forward(self, X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        X: [B,C,H,W]  local feature
        G: [B,1,H,W]  guidance
        Z: [B,C,H,W]
        """
        if X.dim() != 4 or G.dim() != 4:
            raise ValueError("X and G must be BCHW.")
        if G.size(1) != 1:
            raise ValueError("G must be 1-channel: [B,1,H,W].")
        if X.shape[0] != G.shape[0] or X.shape[-2:] != G.shape[-2:]:
            raise ValueError(f"Shape mismatch: X={X.shape}, G={G.shape}")

        r = self.r
        eps = self.eps

        # 1) chi = ||∇G||
        chi = _grad_mag(G, eps=eps)  # [B,1,H,W]
        mu_chi = chi.mean(dim=(-2, -1), keepdim=True)
        eta = self._eta(chi)

        # 2) gamma (use sigmoid for numerical stability)
        gamma = torch.sigmoid(eta * (chi - mu_chi))  # [B,1,H,W]

        # 3) tau (SSIM) using channel-mean of X to align with scalar-map assumption
        Xs = X.mean(dim=1, keepdim=True)             # [B,1,H,W]
        tau_r = _ssim_map(Xs, G, r=r, pad_mode=self.pad_mode)     # [B,1,H,W]
        tau_1 = _ssim_map(Xs, G, r=1, pad_mode=self.pad_mode)     # [B,1,H,W]
        # 关键：把SSIM映射到[0,1]并裁剪，避免负值进入Gamma_S
        tau_r = ((tau_r + 1.0) * 0.5).clamp(0.0, 1.0)
        tau_1 = ((tau_1 + 1.0) * 0.5).clamp(0.0, 1.0)
        tau = tau_r

        # 4) Gamma_G: chi_g = std(G,1) * std(G,r)
        _, var_g1 = _local_var(G, r=1, eps=eps, pad_mode=self.pad_mode)
        _, var_gr = _local_var(G, r=r, eps=eps, pad_mode=self.pad_mode)
        chi_g = torch.sqrt(var_g1) * torch.sqrt(var_gr)  # [B,1,H,W]

        # 5) Gamma_S: nu = tau_1 * tau_r
        #nu = tau_1 * tau_r  # [B,1,H,W]
        nu = (tau_1 * tau_r).clamp_min(0.0)

        # Gamma(x) = (x + c) * mean(1/(x + c))
        inv_mean_chi = (1.0 / (chi_g + eps)).mean(dim=(-2, -1), keepdim=True)
        Gamma_G = (chi_g + eps) * inv_mean_chi  # [B,1,H,W]

        #den_nu = (nu + self.theta).clamp_min(self.eps)  # 防止 0/负数
        #inv_mean_nu = self._inv_mean(den_nu)
        #Gamma_S = den_nu * inv_mean_nu
        inv_mean_nu = (1.0 / (nu + self.theta)).mean(dim=(-2, -1), keepdim=True)
        Gamma_S = (nu + self.theta) * inv_mean_nu  # [B,1,H,W]


        #Gamma_G = Gamma_G.clamp_min(1e-4)
        #Gamma_S = Gamma_S.clamp_min(1e-4)

        # Local linear stats
        muG  = _box_mean(G, r, pad_mode=self.pad_mode)        # [B,1,H,W]
        muX  = _box_mean(X, r, pad_mode=self.pad_mode)        # [B,C,H,W]
        muGX = _box_mean(G * X, r, pad_mode=self.pad_mode)    # [B,C,H,W]
        muG2 = _box_mean(G * G, r, pad_mode=self.pad_mode)    # [B,1,H,W]

        varG  = (muG2 - muG * muG).clamp_min(0.0) + eps       # [B,1,H,W]
        #varG = varG.clamp_min(1e-6)  # 防止 guidance 过平导致 varG 太小
        covGX = muGX - muG * muX                               # [B,C,H,W]

        # Weights
        w_e = self.lam / (Gamma_G + eps)                       # [B,1,H,W]
        w_s = gamma / (Gamma_S + eps)                          # [B,1,H,W]

        #w_e = (self.lam / (Gamma_G + eps)).clamp_max(20.0)  # Gamma_G 已 clamp
        #w_s = (gamma / (Gamma_S + eps)).clamp_max(20.0)  # Gamma_S 已 clamp

        # a,b
        a = (covGX + w_e * gamma + w_s * tau) / (varG + w_e + w_s + eps)  # [B,C,H,W]
        b = muX - a * muG                                      # [B,C,H,W]

        # output aggregation
        mean_a = _box_mean(a, r, pad_mode=self.pad_mode)       # [B,C,H,W]
        mean_b = _box_mean(b, r, pad_mode=self.pad_mode)       # [B,C,H,W]
        Z = mean_a * G + mean_b                                # [B,C,H,W]
        return Z
