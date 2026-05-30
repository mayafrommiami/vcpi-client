"""
MLP architecture for drug → gene expression prediction.

Design choices:
  - Input:  Morgan fingerprint (2048 bits)
  - Output: residual expression (n_genes = 12,995) — per-gene mean added back at inference
  - MC Dropout for uncertainty quantification (keep dropout ON at inference, sample K times)
  - Skip connection from input to penultimate layer (fingerprint identity preserved)
"""

import torch
import torch.nn as nn
import numpy as np


class FingerprintMLP(nn.Module):
    """
    Fingerprint → per-gene expression MLP with residual skip and MC Dropout.

    Architecture:
        fp (2048)
        → Linear(2048, 1024) → BN → GELU → Dropout
        → Linear(1024, 1024) → BN → GELU → Dropout
        → Linear(1024, 512)  → BN → GELU → Dropout
        + skip: Linear(2048, 512)       ← skip from input
        → Linear(512, n_genes)          ← no activation (residuals can be negative)
    """

    def __init__(
        self,
        fp_dim:    int   = 2048,
        n_genes:   int   = 12_995,
        hidden:    list[int] = (1024, 1024, 512),
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.fp_dim  = fp_dim
        self.n_genes = n_genes

        # Main trunk
        layers = []
        in_dim = fp_dim
        for out_dim in hidden:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.trunk = nn.Sequential(*layers)

        # Skip connection from raw fingerprint to penultimate dim
        self.skip = nn.Linear(fp_dim, hidden[-1])

        # Output head
        self.head = nn.Linear(hidden[-1], n_genes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """
        fp: (batch, fp_dim)
        returns: (batch, n_genes) — residuals from per-gene mean
        """
        x = self.trunk(fp) + self.skip(fp)   # skip connection
        return self.head(x)


# ── uncertainty via MC Dropout ─────────────────────────────────────────────────
def mc_predict(
    model: FingerprintMLP,
    fp: torch.Tensor,
    n_samples: int = 30,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout inference: keep dropout ON, sample n_samples times.
    Returns (mean, std) each of shape (batch, n_genes).
    """
    model.train()   # dropout ON
    with torch.no_grad():
        preds = torch.stack(
            [model(fp.to(device)) for _ in range(n_samples)], dim=0
        )  # (n_samples, batch, n_genes)
    model.eval()
    return preds.mean(0), preds.std(0)


# ── wMSE-aware loss ────────────────────────────────────────────────────────────
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE that puts more weight on differentially expressed genes.
    Uses per-gene variance weights (proxy for Mejia weights during training).

    weights: (n_genes,) tensor, sums to 1, higher = more important gene
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        # Normalize to sum to n_genes so scale is comparable to unweighted MSE
        w = weights / weights.sum() * len(weights)
        self.register_buffer("weights", w.float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (batch, n_genes)
        returns: scalar weighted MSE averaged over batch
        """
        sq_err = (pred - target) ** 2          # (batch, n_genes)
        return (sq_err * self.weights).mean()


def build_loss_weights(
    expr_mat: np.ndarray,
    smoothing: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-gene variance weights from training expression matrix.
    expr_mat: (N_compounds, n_genes) float32

    High-variance genes across compounds get higher weight —
    approximates the Mejia weight signal without needing replicate counts.
    """
    var = expr_mat.var(axis=0) + smoothing          # (n_genes,)
    var_norm = var / var.sum()                       # normalize to sum=1
    return torch.from_numpy(var_norm.astype(np.float32))
