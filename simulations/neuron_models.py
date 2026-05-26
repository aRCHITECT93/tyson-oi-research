"""
neuron_models.py
================
Biologically plausible neuron models for organoid intelligence simulation.

Implements:
  - Leaky Integrate-and-Fire (LIF)       — simplest spiking neuron
  - Adaptive Exponential LIF (AdEx)      — captures burst/adaptation like real organoids
  - Hodgkin-Huxley (HH)                  — gold standard biophysical model
  - Izhikevich                           — computationally efficient, biologically rich

Each model is implemented as a pure-numpy/scipy version (CPU, interpretable)
and a snntorch/PyTorch version (GPU-accelerated for large networks).
"""

import numpy as np
from scipy.integrate import odeint
import torch
import snntorch as snn
from snntorch import surrogate
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — neuron parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFParams:
    """Leaky Integrate-and-Fire parameters."""
    tau_m: float = 20.0       # membrane time constant (ms)
    V_rest: float = -70.0     # resting potential (mV)
    V_thresh: float = -55.0   # spike threshold (mV)
    V_reset: float = -75.0    # reset potential after spike (mV)
    R_m: float = 10.0         # membrane resistance (MOhm)
    t_refrac: float = 2.0     # refractory period (ms)


@dataclass
class AdExParams:
    """Adaptive Exponential LIF — captures organoid burst dynamics."""
    C_m: float = 200.0        # membrane capacitance (pF)
    g_L: float = 10.0         # leak conductance (nS)
    E_L: float = -70.0        # leak reversal potential (mV)
    V_T: float = -50.0        # threshold slope factor (mV)
    delta_T: float = 2.0      # sharpness of action potential (mV)
    V_thresh: float = 20.0    # spike cut-off (mV)
    V_reset: float = -58.0    # reset potential (mV)
    a: float = 2.0            # subthreshold adaptation (nS)
    b: float = 100.0          # spike-triggered adaptation (pA)
    tau_w: float = 30.0       # adaptation time constant (ms)


@dataclass
class IzhikevichParams:
    """
    Izhikevich neuron — 4 parameters cover 20+ biological neuron types.
    Common presets:
      Regular spiking (RS):    a=0.02, b=0.2,  c=-65, d=8
      Fast spiking (FS):       a=0.1,  b=0.2,  c=-65, d=2
      Bursting (IB):           a=0.02, b=0.2,  c=-55, d=4
      Chattering (CH):         a=0.02, b=0.2,  c=-50, d=2   ← closest to organoid
      Low-threshold (LTS):     a=0.02, b=0.25, c=-65, d=2
    """
    a: float = 0.02
    b: float = 0.2
    c: float = -50.0
    d: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY IMPLEMENTATIONS (single neuron, interpretable)
# ─────────────────────────────────────────────────────────────────────────────

class LIFNeuron:
    """Leaky Integrate-and-Fire — the workhorse of computational neuro."""

    def __init__(self, params: Optional[LIFParams] = None, dt: float = 0.1):
        self.p = params or LIFParams()
        self.dt = dt
        self.V = self.p.V_rest
        self.t_since_spike = 999.0  # ms since last spike (starts large = not refractory)

    def step(self, I_ext: float) -> Tuple[float, bool]:
        """
        Advance one time step.
        Returns: (membrane_voltage, spiked_bool)
        """
        spiked = False

        if self.t_since_spike < self.p.t_refrac:
            # Refractory — clamp to reset
            self.V = self.p.V_reset
            self.t_since_spike += self.dt
            return self.V, spiked

        # Leak + input
        dV = (-(self.V - self.p.V_rest) + self.p.R_m * I_ext) / self.p.tau_m
        self.V += dV * self.dt

        if self.V >= self.p.V_thresh:
            self.V = self.p.V_reset
            self.t_since_spike = 0.0
            spiked = True
        else:
            self.t_since_spike += self.dt

        return self.V, spiked

    def simulate(self, I_trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run full simulation over a current trace. Returns (V_trace, spike_train)."""
        V_trace = np.zeros(len(I_trace))
        spikes = np.zeros(len(I_trace), dtype=bool)
        for i, I in enumerate(I_trace):
            V_trace[i], spikes[i] = self.step(I)
        return V_trace, spikes


class AdExNeuron:
    """
    Adaptive Exponential LIF.
    The adaptation variable w captures the history-dependence seen in organoids
    — neurons that fired recently are harder to fire again (spike-frequency adaptation).
    """

    def __init__(self, params: Optional[AdExParams] = None, dt: float = 0.1):
        self.p = params or AdExParams()
        self.dt = dt
        self.V = self.p.E_L
        self.w = 0.0   # adaptation current (pA)

    def step(self, I_ext: float) -> Tuple[float, float, bool]:
        """Returns (V, w, spiked)."""
        spiked = False
        p = self.p

        # AdEx equations
        I_leak = p.g_L * (p.E_L - self.V)
        I_exp = p.g_L * p.delta_T * np.exp((self.V - p.V_T) / p.delta_T)
        dV = (I_leak + I_exp - self.w + I_ext) / p.C_m
        dw = (p.a * (self.V - p.E_L) - self.w) / p.tau_w

        self.V += dV * self.dt * 1000  # convert to mV/ms
        self.w += dw * self.dt * 1000

        if self.V >= p.V_thresh:
            self.V = p.V_reset
            self.w += p.b
            spiked = True

        return self.V, self.w, spiked

    def simulate(self, I_trace: np.ndarray):
        V_trace = np.zeros(len(I_trace))
        w_trace = np.zeros(len(I_trace))
        spikes = np.zeros(len(I_trace), dtype=bool)
        for i, I in enumerate(I_trace):
            V_trace[i], w_trace[i], spikes[i] = self.step(I)
        return V_trace, w_trace, spikes


class IzhikevichNeuron:
    """
    Izhikevich model — best balance of biological realism and speed.
    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30mV: v = c, u = u + d
    """

    def __init__(self, params: Optional[IzhikevichParams] = None, dt: float = 0.5):
        self.p = params or IzhikevichParams()
        self.dt = dt
        self.v = -65.0
        self.u = self.p.b * self.v

    def step(self, I: float) -> Tuple[float, bool]:
        spiked = False
        p = self.p

        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * self.dt
        du = (p.a * (p.b * self.v - self.u)) * self.dt
        self.v += dv
        self.u += du

        if self.v >= 30.0:
            self.v = p.c
            self.u += p.d
            spiked = True

        return self.v, spiked

    def simulate(self, I_trace: np.ndarray):
        V = np.zeros(len(I_trace))
        spikes = np.zeros(len(I_trace), dtype=bool)
        for i, I in enumerate(I_trace):
            V[i], spikes[i] = self.step(I)
        return V, spikes


# ─────────────────────────────────────────────────────────────────────────────
# HODGKIN-HUXLEY (full biophysical, for validation / theory work)
# ─────────────────────────────────────────────────────────────────────────────

def hodgkin_huxley(I_ext_func, t_span: np.ndarray) -> np.ndarray:
    """
    Full Hodgkin-Huxley model via ODE integration.
    Returns array of shape (len(t_span), 4) — [V, m, h, n]

    V: membrane voltage (mV)
    m, h: sodium channel gating variables
    n:    potassium channel gating variable
    """
    # Conductances (mS/cm²) and Nernst potentials (mV)
    g_Na, g_K, g_L = 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.387
    C_m = 1.0  # uF/cm²

    def alpha_m(V): return 0.1*(V+40)/(1-np.exp(-(V+40)/10)+1e-7)
    def beta_m(V):  return 4*np.exp(-(V+65)/18)
    def alpha_h(V): return 0.07*np.exp(-(V+65)/20)
    def beta_h(V):  return 1/(1+np.exp(-(V+35)/10))
    def alpha_n(V): return 0.01*(V+55)/(1-np.exp(-(V+55)/10)+1e-7)
    def beta_n(V):  return 0.125*np.exp(-(V+65)/80)

    def hh_odes(y, t):
        V, m, h, n = y
        I = I_ext_func(t)
        dVdt = (I - g_Na*m**3*h*(V-E_Na) - g_K*n**4*(V-E_K) - g_L*(V-E_L)) / C_m
        dmdt = alpha_m(V)*(1-m) - beta_m(V)*m
        dhdt = alpha_h(V)*(1-h) - beta_h(V)*h
        dndt = alpha_n(V)*(1-n) - beta_n(V)*n
        return [dVdt, dmdt, dhdt, dndt]

    # Steady-state initial conditions
    V0 = -65.0
    y0 = [V0,
          alpha_m(V0)/(alpha_m(V0)+beta_m(V0)),
          alpha_h(V0)/(alpha_h(V0)+beta_h(V0)),
          alpha_n(V0)/(alpha_n(V0)+beta_n(V0))]

    return odeint(hh_odes, y0, t_span)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-ACCELERATED LIF POPULATION (snntorch + PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class GPUNeuronPopulation(torch.nn.Module):
    """
    Large population of LIF neurons running on the RTX 5070 Ti.
    Uses snntorch's surrogate gradient for differentiability.

    Suitable for populations of 10k–1M neurons that would be
    too slow in pure Python.
    """

    def __init__(self, n_neurons: int = 1000,
                 beta: float = 0.9,
                 threshold: float = 1.0,
                 device: Optional[str] = None):
        super().__init__()
        self.n = n_neurons
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Surrogate gradient function — needed for backprop through spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            learn_beta=True,        # let the network learn its own time constants
            learn_threshold=True    # and thresholds
        ).to(self.device)

        self.mem = self.lif.init_leaky()

    def forward(self, input_spikes: torch.Tensor):
        """
        input_spikes: (batch, n_neurons) float tensor
        Returns: (output_spikes, membrane_potential)
        """
        spk, mem = self.lif(input_spikes, self.mem)
        self.mem = mem
        return spk, mem

    def reset(self):
        self.mem = self.lif.init_leaky()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO / SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.1   # ms
    T  = 500   # ms
    t  = np.arange(0, T, dt)

    # Step current: 0 for first 100ms, then 3.5 nA
    I = np.where(t > 100, 3.5, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Neuron Model Comparison — Organoid Intelligence Foundation", fontsize=14)

    # LIF
    lif = LIFNeuron(dt=dt)
    V_lif, sp_lif = lif.simulate(I)
    axes[0].plot(t, V_lif, color='steelblue', lw=0.8)
    axes[0].set_title(f"LIF — {sp_lif.sum()} spikes")
    axes[0].set_ylabel("mV"); axes[0].axvline(100, color='red', ls='--', alpha=0.4)

    # AdEx (burst-capable, closest to organoid behavior)
    adex = AdExNeuron(dt=dt)
    V_adex, w_adex, sp_adex = adex.simulate(I * 500)  # AdEx uses pA
    axes[1].plot(t, V_adex, color='coral', lw=0.8)
    axes[1].set_title(f"AdEx (organoid-like bursting) — {sp_adex.sum()} spikes")
    axes[1].set_ylabel("mV"); axes[1].axvline(100, color='red', ls='--', alpha=0.4)

    # Izhikevich chattering (CH preset)
    izh = IzhikevichNeuron(IzhikevichParams(a=0.02, b=0.2, c=-50.0, d=2.0), dt=0.5)
    t2 = np.arange(0, T, 0.5)
    I2 = np.where(t2 > 100, 10.0, 0.0)
    V_izh, sp_izh = izh.simulate(I2)
    axes[2].plot(t2, V_izh, color='mediumseagreen', lw=0.8)
    axes[2].set_title(f"Izhikevich CH (chattering) — {sp_izh.sum()} spikes")
    axes[2].set_ylabel("mV"); axes[2].set_xlabel("Time (ms)")
    axes[2].axvline(100, color='red', ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("experiments/results/neuron_models_comparison.png", dpi=150)
    print("Saved: experiments/results/neuron_models_comparison.png")
    plt.show()

    # GPU population test
    if torch.cuda.is_available():
        pop = GPUNeuronPopulation(n_neurons=10000)
        x = torch.rand(10000).to(pop.device)
        spk, mem = pop(x)
        print(f"\nGPU Population (10k neurons): {int(spk.sum())} spikes fired | device: {pop.device}")
    else:
        print("\nCUDA not available — GPU population test skipped")
