"""Pure-PyTorch fallback for `mamba_ssm.ops.selective_scan_interface`.

This is a compatibility implementation designed for environments where the
optimized CUDA/Triton kernels from `mamba_ssm` are unavailable.

The function signature mirrors the commonly used API:

    selective_scan_fn(u, delta, A, B, C, D, z=None,
                      delta_bias=None, delta_softplus=False,
                      return_last_state=None)

Notes:
- This implementation prioritizes correctness/compatibility over speed.
- Runtime is significantly slower than the official optimized package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool | None = None,
):
    """Fallback selective scan.

    Args:
        u: Input signal, shape ``(batch, d_inner, seq_len)``.
        delta: Per-token time-step values, shape ``(batch, d_inner, seq_len)``.
        A: State transition log-parameter (usually negative), shape ``(d_inner, d_state)``.
        B: Input projection to state, shape ``(batch, d_state, seq_len)``.
        C: State projection to output, shape ``(batch, d_state, seq_len)``.
        D: Skip/diagonal term, shape ``(d_inner,)``.
        z: Optional gating tensor of shape ``(batch, d_inner, seq_len)``.
        delta_bias: Optional bias term added to ``delta`` (typically shape ``(d_inner,)``).
        delta_softplus: If True, apply ``softplus`` to ``delta`` after bias.
        return_last_state: If truthy, return tuple ``(y, last_state)``.

    Returns:
        ``y`` with shape ``(batch, d_inner, seq_len)``, or ``(y, state)`` if
        ``return_last_state`` is truthy where ``state`` is
        ``(batch, d_inner, d_state)``.
    """
    if u.ndim != 3:
        raise ValueError(f"Expected u with 3 dims (B, D, L), got shape {tuple(u.shape)}")

    batch, d_inner, seq_len = u.shape

    if delta.shape != (batch, d_inner, seq_len):
        raise ValueError(
            f"delta shape mismatch: expected {(batch, d_inner, seq_len)}, got {tuple(delta.shape)}"
        )

    if A.ndim != 2 or A.shape[0] != d_inner:
        raise ValueError(f"A shape mismatch: expected (D, N) with D={d_inner}, got {tuple(A.shape)}")

    d_state = A.shape[1]

    if B.shape != (batch, d_state, seq_len):
        raise ValueError(
            f"B shape mismatch: expected {(batch, d_state, seq_len)}, got {tuple(B.shape)}"
        )
    if C.shape != (batch, d_state, seq_len):
        raise ValueError(
            f"C shape mismatch: expected {(batch, d_state, seq_len)}, got {tuple(C.shape)}"
        )
    if D.ndim != 1 or D.shape[0] != d_inner:
        raise ValueError(f"D shape mismatch: expected ({d_inner},), got {tuple(D.shape)}")

    # Work in the input dtype/device for compatibility with surrounding model code.
    delta_eff = delta
    if delta_bias is not None:
        # Broadcast common bias shapes: (D,), (1, D), (D, 1)
        if delta_bias.ndim == 1:
            delta_eff = delta_eff + delta_bias.view(1, -1, 1)
        elif delta_bias.ndim == 2 and delta_bias.shape[0] == 1:
            delta_eff = delta_eff + delta_bias.view(1, -1, 1)
        elif delta_bias.ndim == 2 and delta_bias.shape[1] == 1:
            delta_eff = delta_eff + delta_bias.view(1, -1, 1)
        else:
            # Fall back to generic broadcast if possible.
            delta_eff = delta_eff + delta_bias

    if delta_softplus:
        delta_eff = F.softplus(delta_eff)

    # Recurrent state: (B, D, N)
    state = torch.zeros(
        batch,
        d_inner,
        d_state,
        device=u.device,
        dtype=u.dtype,
    )

    y_chunks = []

    # Selective scan recurrence
    # state_t = exp(delta_t * A) * state_{t-1} + (delta_t * B_t) * u_t
    # y_t     = <state_t, C_t> + D * u_t
    for t in range(seq_len):
        u_t = u[:, :, t]  # (B, D)
        dt_t = delta_eff[:, :, t]  # (B, D)
        b_t = B[:, :, t]  # (B, N)
        c_t = C[:, :, t]  # (B, N)

        # (B, D, N)
        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)

        state = dA * state + dB * u_t.unsqueeze(-1)

        y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + D.view(1, -1) * u_t
        y_chunks.append(y_t)

    y = torch.stack(y_chunks, dim=-1)  # (B, D, L)

    if z is not None:
        # Common gating pattern in SSM blocks.
        y = y * F.silu(z)

    if return_last_state:
        return y, state
    return y


__all__ = ["selective_scan_fn"]
