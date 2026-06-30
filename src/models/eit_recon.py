"""Learned EIT reconstruction (channels -> conductivity image).

A sub-track of the foundation work: instead of (or alongside) the existing
1-step-Newton 40x40 image, learn a decoder from the raw ring channels
(resistance + reactance) to a conductivity image. This can be trained:

  * supervised against a (finer) reference reconstruction, and/or
  * physics-regularized via a differentiable forward operator that maps the
    reconstructed image back to boundary measurements (data-consistency), and/or
  * jointly with the SSL/BP objective (a reconstruction maximally predictive of
    dynamics).

The real EIT geometry / forward model is not bundled (see PLAN.md "Remaining
inputs"); `EITForwardOperator` therefore defaults to a learnable linear operator
(a data-driven stand-in) and can be swapped for a fixed physics operator by
passing its measurement matrix.
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)


def circular_fov_mask(height: int, width: int = None,
                      radius: float = None) -> torch.Tensor:
    """Boolean (H, W) mask, True inside the circular field of view.

    Mirrors the ring's fixed circular FOV (constant NaN region outside the disc).
    """
    width = width or height
    radius = radius if radius is not None else min(height, width) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


class EITReconstructor(nn.Module):
    """Decode ring channels (B, C, T) -> conductivity image (B, out, H, W, T)."""

    def __init__(self,
                 in_channels: int,
                 img_size: int = 40,
                 hidden: int = 256,
                 num_hidden_layers: int = 2,
                 out_channels: int = 1,
                 apply_fov: bool = True,
                 refine: bool = True,
                 ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.img_size = int(img_size)
        self.out_channels = int(out_channels)

        layers = [nn.Linear(self.in_channels, hidden), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, self.out_channels * self.img_size * self.img_size)]
        self.decoder = nn.Sequential(*layers)

        # optional spatial refinement of the decoded image
        self.refine = (
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            if refine else None
        )

        if apply_fov:
            self.register_buffer("fov", circular_fov_mask(self.img_size).to(torch.float32))
        else:
            self.fov = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        H = W = self.img_size

        xt = x.permute(0, 2, 1).reshape(B * T, C)          # (B*T, C)
        img = self.decoder(xt).view(B * T, self.out_channels, H, W)

        if self.refine is not None:
            img = self.refine(img)

        if self.fov is not None:
            img = img * self.fov                            # zero outside the disc

        img = img.view(B, T, self.out_channels, H, W)
        return img.permute(0, 2, 3, 4, 1).contiguous()      # (B, out, H, W, T)


class EITForwardOperator(nn.Module):
    """Differentiable image -> boundary-measurement operator (data-consistency).

    Defaults to a *learnable* linear operator (data-driven stand-in). To use a
    fixed physics forward model, pass its measurement matrix `weight`
    (shape: (num_measurements, H*W)) and set `learnable=False`.
    """

    def __init__(self,
                 img_size: int = 40,
                 num_measurements: int = 64,
                 weight: torch.Tensor = None,
                 learnable: bool = True,
                 ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.num_measurements = int(num_measurements)
        n_pix = self.img_size * self.img_size

        if weight is None:
            weight = torch.randn(self.num_measurements, n_pix) / (n_pix ** 0.5)
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            if weight.shape != (self.num_measurements, n_pix):
                raise ValueError(f"weight must be ({self.num_measurements}, {n_pix}); got {tuple(weight.shape)}")

        if learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 1, H, W, T) -> voltages (B, num_measurements, T)
        B, _, H, W, T = img.shape
        flat = img.reshape(B, H * W, T)                     # (B, H*W, T)
        v = torch.einsum("mp,bpt->bmt", self.weight, flat)  # (B, M, T)
        return v


def reconstruction_loss(pred_img: torch.Tensor,
                        target_img: torch.Tensor,
                        fov: torch.Tensor = None) -> torch.Tensor:
    """Supervised reconstruction MSE, optionally restricted to the FOV."""
    if fov is not None:
        fov = fov.to(dtype=pred_img.dtype)
        # broadcast (H, W) mask over (B, out, H, W, T) and average over FOV pixels
        diff = (pred_img - target_img) * fov[None, None, :, :, None]
        denom = fov.sum().clamp_min(1) * pred_img.shape[0] * pred_img.shape[1] * pred_img.shape[-1]
        return diff.square().sum() / denom
    return nn.functional.mse_loss(pred_img, target_img)


def data_consistency_loss(pred_img: torch.Tensor,
                          measured: torch.Tensor,
                          operator: EITForwardOperator) -> torch.Tensor:
    """MSE between the forward-projected reconstruction and measured voltages."""
    return nn.functional.mse_loss(operator(pred_img), measured)
