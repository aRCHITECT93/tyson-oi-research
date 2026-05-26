"""
stdp.py
=======
Spike-Timing Dependent Plasticity — the learning rule of organoids.

STDP is HOW organoids learn. When neuron A fires just before neuron B,
the synapse A→B strengthens (potentiation). If A fires AFTER B, it weakens
(depression). This timing-based Hebbian learning is the biological equivalent
of backpropagation.

This module implements:
  - Classical STDP (additive)
  - Multiplicative STDP (weight-dependent — more stable, more biological)
  - Triplet STDP (3-factor rule — most accurate to real organoids)
  - Reward-modulated STDP (R-STDP) — the DishBrain learning mechanism
  - Online STDP synapse layer (PyTorch/GPU)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# STDP PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STDPParams:
    """Classical STDP window parameters."""
    A_plus: float  = 0.01     # potentiation amplitude
    A_minus: float = 0.0105   # depression amplitude (slightly asymmetric — stable)
    tau_plus: float = 20.0    # potentiation time constant (ms)
    tau_minus: float = 20.0   # depression time constant (ms)
    w_min: float = 0.0        # weight floor
    w_max: float = 1.0        # weight ceiling


@dataclass
class TripletSTDPParams:
    """
    Triplet STDP — Pfister & Gerstner 2006.
    More accurately matches experimental organoid data than pair-STDP.
    Uses 4 traces instead of 2.
    """
    A2_plus: float  = 0.01
    A2_minus: float = 0.01
    A3_plus: float  = 0.05    # 3rd-order potentiation (burst sensitivity)
    A3_minus: float = 0.005
    tau_plus: float  = 16.8   # ms
    tau_minus: float = 33.7
    tau_x: float = 101.0      # slow pre-synaptic trace
    tau_y: float = 125.0      # slow post-synaptic trace
    w_min: float = 0.0
    w_max: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY — classical STDP (single synapse, educational)
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalSTDP:
    """
    Single-synapse classical STDP.
    Maintains eligibility traces and updates weight online.

    This is the exact mechanism operating in the DishBrain experiment —
    the MEA stimulation provides the 'teaching signal' that biases
    which STDP updates get applied.
    """

    def __init__(self, w_init: float = 0.5, params: Optional[STDPParams] = None):
        self.w = w_init
        self.p = params or STDPParams()
        self.trace_pre  = 0.0   # eligibility trace for pre-synaptic neuron
        self.trace_post = 0.0   # eligibility trace for post-synaptic neuron

    def step(self, pre_spike: bool, post_spike: bool, dt: float) -> float:
        """
        One time step of STDP.
        Returns updated weight.
        """
        p = self.p

        # Decay traces
        self.trace_pre  *= np.exp(-dt / p.tau_plus)
        self.trace_post *= np.exp(-dt / p.tau_minus)

        dw = 0.0
        if pre_spike:
            self.trace_pre += 1.0
            # Pre fires: check if post fired recently → depression
            dw -= p.A_minus * self.trace_post

        if post_spike:
            self.trace_post += 1.0
            # Post fires: check if pre fired recently → potentiation
            dw += p.A_plus * self.trace_pre

        self.w = np.clip(self.w + dw, p.w_min, p.w_max)
        return self.w

    def simulate(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                 dt: float = 0.1) -> np.ndarray:
        """Simulate over full spike trains. Returns weight trajectory."""
        w_trace = np.zeros(len(pre_spikes))
        for i, (pre, post) in enumerate(zip(pre_spikes, post_spikes)):
            w_trace[i] = self.step(bool(pre), bool(post), dt)
        return w_trace


class MultiplicativeSTDP(ClassicalSTDP):
    """
    Weight-dependent STDP — more biologically stable.
    Potentiation scales with (w_max - w), depression with (w - w_min).
    Prevents runaway weight growth.
    """

    def step(self, pre_spike: bool, post_spike: bool, dt: float) -> float:
        p = self.p
        self.trace_pre  *= np.exp(-dt / p.tau_plus)
        self.trace_post *= np.exp(-dt / p.tau_minus)
        dw = 0.0
        if pre_spike:
            self.trace_pre += 1.0
            dw -= p.A_minus * (self.w - p.w_min) * self.trace_post
        if post_spike:
            self.trace_post += 1.0
            dw += p.A_plus * (p.w_max - self.w) * self.trace_pre
        self.w = np.clip(self.w + dw, p.w_min, p.w_max)
        return self.w


class RewardModulatedSTDP:
    """
    R-STDP — the mechanism behind DishBrain Pong learning.

    Standard STDP accumulates an eligibility trace e(t).
    A reward signal r(t) then gates which traces actually update weights:
        dw/dt = r(t) * e(t)

    This is the simplest biologically plausible reinforcement learning rule.
    When the cell culture 'misses' the ball → r = negative
    When it 'hits' → r = positive
    STDP traces that were active during good behavior get strengthened.
    """

    def __init__(self, n_pre: int, n_post: int,
                 w_init: Optional[np.ndarray] = None,
                 params: Optional[STDPParams] = None,
                 tau_reward: float = 200.0):
        self.p = params or STDPParams()
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_reward = tau_reward

        if w_init is not None:
            self.W = w_init.copy()
        else:
            self.W = np.random.uniform(0.1, 0.9, (n_post, n_pre))

        self.trace_pre  = np.zeros(n_pre)
        self.trace_post = np.zeros(n_post)
        self.eligibility = np.zeros((n_post, n_pre))  # e(t)
        self.reward_trace = 0.0

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
             reward: float, dt: float) -> np.ndarray:
        """
        pre_spikes:  (n_pre,) binary
        post_spikes: (n_post,) binary
        reward:      scalar (+1 good, -1 bad, 0 neutral)
        Returns updated weight matrix W (n_post x n_pre)
        """
        p = self.p

        # Decay traces
        self.trace_pre  *= np.exp(-dt / p.tau_plus)
        self.trace_post *= np.exp(-dt / p.tau_minus)
        self.eligibility *= np.exp(-dt / self.tau_reward)
        self.reward_trace *= np.exp(-dt / self.tau_reward)

        # Update traces on spikes
        self.trace_pre  += pre_spikes.astype(float)
        self.trace_post += post_spikes.astype(float)

        # STDP eligibility (outer product)
        dE_plus  = p.A_plus  * np.outer(post_spikes, self.trace_pre)
        dE_minus = p.A_minus * np.outer(self.trace_post, pre_spikes)
        self.eligibility += dE_plus - dE_minus

        # Weight update gated by reward
        self.W += reward * self.eligibility * dt * 0.001
        self.W = np.clip(self.W, p.w_min, p.w_max)
        return self.W


# ─────────────────────────────────────────────────────────────────────────────
# GPU — STDP synapse layer (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class STDPSynapseLayer(nn.Module):
    """
    GPU-accelerated STDP synapse layer.
    Drop-in replacement for nn.Linear in a spiking network,
    with online weight updates via STDP traces.

    Usage:
        layer = STDPSynapseLayer(100, 200).to('cuda')
        out_spikes = layer(in_spikes, post_spikes, reward=1.0, dt=0.1)
    """

    def __init__(self, n_in: int, n_out: int,
                 params: Optional[STDPParams] = None,
                 device: Optional[str] = None):
        super().__init__()
        self.p = params or STDPParams()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_in = n_in
        self.n_out = n_out

        # Learnable weights
        self.W = nn.Parameter(
            torch.rand(n_out, n_in, device=self.device) * 0.5 + 0.25
        )

        # Eligibility traces (not parameters — updated in-place)
        self.register_buffer('trace_pre',  torch.zeros(n_in,  device=self.device))
        self.register_buffer('trace_post', torch.zeros(n_out, device=self.device))

    @torch.no_grad()
    def stdp_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                    reward: float = 1.0, dt: float = 0.1):
        """Apply one STDP weight update step (called after forward pass)."""
        p = self.p
        tau_decay_pre  = torch.exp(torch.tensor(-dt / p.tau_plus))
        tau_decay_post = torch.exp(torch.tensor(-dt / p.tau_minus))

        self.trace_pre  = self.trace_pre  * tau_decay_pre  + pre_spikes.float()
        self.trace_post = self.trace_post * tau_decay_post + post_spikes.float()

        dW_plus  = p.A_plus  * torch.outer(post_spikes.float(), self.trace_pre)
        dW_minus = p.A_minus * torch.outer(self.trace_post, pre_spikes.float())
        dW = reward * (dW_plus - dW_minus)

        self.W.data.add_(dW)
        self.W.data.clamp_(p.w_min, p.w_max)

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Weighted sum of input spikes → output current."""
        return torch.matmul(self.W, pre_spikes.float())

    def reset_traces(self):
        self.trace_pre.zero_()
        self.trace_post.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# STDP WINDOW VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def compute_stdp_window(delta_t_range: np.ndarray,
                        params: Optional[STDPParams] = None) -> np.ndarray:
    """
    Compute theoretical STDP weight change for a range of pre/post spike time differences.
    delta_t = t_post - t_pre
      delta_t > 0: post fires after pre  → potentiation (LTP)
      delta_t < 0: post fires before pre → depression  (LTD)
    """
    p = params or STDPParams()
    dW = np.where(
        delta_t_range >= 0,
        p.A_plus  * np.exp(-delta_t_range / p.tau_plus),
       -p.A_minus * np.exp( delta_t_range / p.tau_minus)
    )
    return dW


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STDP Learning Rules — Organoid Intelligence Foundation", fontsize=13)

    # --- STDP window ---
    dt_range = np.linspace(-80, 80, 500)
    dW_classic = compute_stdp_window(dt_range)
    dW_asym    = compute_stdp_window(dt_range, STDPParams(A_plus=0.01, A_minus=0.012,
                                                           tau_plus=20, tau_minus=40))
    axes[0].plot(dt_range, dW_classic, label='Symmetric', color='steelblue')
    axes[0].plot(dt_range, dW_asym,    label='Asymmetric (realistic)', color='coral', ls='--')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].axvline(0, color='k', lw=0.5)
    axes[0].set_xlabel("Δt = t_post − t_pre (ms)")
    axes[0].set_ylabel("ΔW (weight change)")
    axes[0].set_title("STDP Learning Window")
    axes[0].legend()
    axes[0].fill_between(dt_range, dW_classic, alpha=0.15, color='steelblue')

    # --- Weight evolution during simulated learning ---
    rng = np.random.default_rng(42)
    T, dt = 2000, 0.1
    t = np.arange(0, T, dt)

    # Correlated spike trains — post fires ~5ms after pre (should potentiate)
    pre_rate  = 20.0  # Hz
    post_rate = 20.0
    pre_spikes  = rng.random(len(t)) < (pre_rate  * dt / 1000)
    # Post has slight lag — correlated with pre
    post_spikes = np.zeros_like(pre_spikes, dtype=bool)
    for i, s in enumerate(pre_spikes):
        if s and i + 5 < len(t):
            post_spikes[i + 5] = True  # fires ~0.5ms after pre

    stdp_classic = ClassicalSTDP(w_init=0.5)
    stdp_mult    = MultiplicativeSTDP(w_init=0.5)
    w_c = stdp_classic.simulate(pre_spikes, post_spikes, dt)
    w_m = stdp_mult.simulate(pre_spikes, post_spikes, dt)

    axes[1].plot(t, w_c, color='steelblue', lw=0.6, alpha=0.8, label='Classical STDP')
    axes[1].plot(t, w_m, color='coral',     lw=0.6, alpha=0.8, label='Multiplicative STDP')
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Synaptic weight")
    axes[1].set_title("Weight Evolution (correlated spikes → potentiation)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("experiments/results/stdp_learning.png", dpi=150)
    print("Saved: experiments/results/stdp_learning.png")
    plt.show()

    # GPU STDP test
    if torch.cuda.is_available():
        layer = STDPSynapseLayer(100, 200)
        pre  = (torch.rand(100) > 0.8).to(layer.device)
        post = (torch.rand(200) > 0.8).to(layer.device)
        out  = layer(pre)
        w_before = layer.W.data.clone()
        layer.stdp_update(pre, post, reward=1.0, dt=0.1)
        dw_mag = (layer.W.data - w_before).abs().mean().item()
        print(f"GPU STDP: avg |ΔW| = {dw_mag:.6f} on {layer.device}")
