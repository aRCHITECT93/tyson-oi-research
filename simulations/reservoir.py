"""
reservoir.py
============
Reservoir Computing — the theoretical framework for organoid intelligence.

Why reservoir computing (RC) maps perfectly to organoids:
  1. Organoids are random, recurrently connected networks (like a reservoir)
  2. They're not trained end-to-end — only the readout layer learns
  3. The biological substrate provides the 'liquid' state transformation
  4. Input stimulation drives the reservoir; MEA reads out the state

This module implements:
  - Echo State Network (ESN)         — classical liquid state machine
  - Spiking Reservoir (LSM)          — biological version with LIF neurons
  - OrganoidReservoir                — the OI-specific model (R-STDP + AdEx)
  - ReservoirAnalyzer                — tools to measure reservoir quality
    (memory capacity, separation property, echo state property)

The OrganoidReservoir is the core theoretical contribution:
  it models the MEA-organoid interface as a reservoir computing system
  with biologically constrained dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from .neuron_models import IzhikevichNeuron, IzhikevichParams, AdExNeuron, AdExParams
from .stdp import RewardModulatedSTDP, STDPParams


# ─────────────────────────────────────────────────────────────────────────────
# ECHO STATE NETWORK (classical reservoir — baseline)
# ─────────────────────────────────────────────────────────────────────────────

class EchoStateNetwork:
    """
    Classical Echo State Network.
    Uses a random recurrent reservoir with fixed weights.
    Only the readout layer (W_out) is trained via ridge regression.

    This is the computational model we compare organoids against.
    Key question: does biological noise HELP or HURT reservoir quality?
    """

    def __init__(self,
                 n_inputs: int,
                 n_reservoir: int = 500,
                 n_outputs: int = 1,
                 spectral_radius: float = 0.9,   # < 1.0 for echo state property
                 sparsity: float = 0.9,           # fraction of zero connections
                 input_scaling: float = 0.5,
                 noise_level: float = 0.001,
                 leak_rate: float = 0.3,          # temporal filtering (0=fast, 1=slow)
                 seed: int = 42):

        self.n_in = n_inputs
        self.n_res = n_reservoir
        self.n_out = n_outputs
        self.leak = leak_rate
        self.noise = noise_level
        self.rng = np.random.default_rng(seed)

        # Input weights (fixed)
        self.W_in = self.rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_inputs))

        # Reservoir weights (fixed, sparse, scaled to spectral radius)
        W = self.rng.standard_normal((n_reservoir, n_reservoir))
        mask = self.rng.random((n_reservoir, n_reservoir)) < sparsity
        W[mask] = 0.0
        eigenvalues = np.linalg.eigvals(W)
        self.W_res = W * (spectral_radius / np.max(np.abs(eigenvalues)))

        # Readout weights (trained)
        self.W_out = None

        # State
        self.state = np.zeros(n_reservoir)

    def _step(self, u: np.ndarray) -> np.ndarray:
        """One reservoir step."""
        pre = self.W_in @ u + self.W_res @ self.state
        pre += self.rng.normal(0, self.noise, self.n_res)
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)
        return self.state.copy()

    def run(self, inputs: np.ndarray, washout: int = 100) -> np.ndarray:
        """
        Drive reservoir with input sequence.
        Returns state matrix (T - washout, n_reservoir).
        """
        self.state = np.zeros(self.n_res)
        states = []
        for t, u in enumerate(inputs):
            s = self._step(u)
            if t >= washout:
                states.append(s)
        return np.array(states)

    def train(self, inputs: np.ndarray, targets: np.ndarray,
              washout: int = 100, ridge: float = 1e-6):
        """Train readout via ridge regression."""
        states = self.run(inputs, washout)
        T = targets[washout:]
        # Ridge: W_out = T @ S^T (S S^T + λI)^-1
        self.W_out = np.linalg.solve(
            states.T @ states + ridge * np.eye(self.n_res),
            states.T @ T
        ).T

    def predict(self, inputs: np.ndarray, washout: int = 100) -> np.ndarray:
        states = self.run(inputs, washout)
        return states @ self.W_out.T


# ─────────────────────────────────────────────────────────────────────────────
# LIQUID STATE MACHINE — spiking reservoir
# ─────────────────────────────────────────────────────────────────────────────

class LiquidStateMachine:
    """
    Liquid State Machine (LSM) — the spiking version of ESN.
    Uses Izhikevich neurons for biological realism.

    Organoids ARE LSMs. This is the direct computational model.
    The MEA chip provides:
      - Input: electrical stimulation patterns → W_in encoding
      - Output: spike population readout → W_out decoding
    """

    def __init__(self,
                 n_inputs: int,
                 n_neurons: int = 200,
                 n_outputs: int = 1,
                 connection_prob: float = 0.1,
                 w_ee: float = 0.5,   # excitatory→excitatory weight
                 w_ei: float = 0.7,   # excitatory→inhibitory
                 w_ie: float = -1.0,  # inhibitory→excitatory
                 w_ii: float = -0.5,  # inhibitory→inhibitory
                 exc_fraction: float = 0.8,  # Dale's law
                 dt: float = 0.5,
                 seed: int = 42):

        self.n_in   = n_inputs
        self.n      = n_neurons
        self.n_out  = n_outputs
        self.dt     = dt
        self.rng    = np.random.default_rng(seed)

        # Neuron types (Dale's law — biological constraint)
        n_exc = int(n_neurons * exc_fraction)
        n_inh = n_neurons - n_exc
        self.is_exc = np.array([True]*n_exc + [False]*n_inh)

        # Create neurons
        self.neurons = []
        for exc in self.is_exc:
            if exc:
                p = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)   # RS
            else:
                p = IzhikevichParams(a=0.1,  b=0.2, c=-65.0, d=2.0)   # FS
            self.neurons.append(IzhikevichNeuron(p, dt=dt))

        # Input weights
        self.W_in = self.rng.uniform(5, 15, (n_neurons, n_inputs))

        # Recurrent weights (sparse, typed by Dale's law)
        self.W_rec = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and self.rng.random() < connection_prob:
                    if self.is_exc[j]:
                        self.W_rec[i,j] = w_ee if self.is_exc[i] else w_ei
                    else:
                        self.W_rec[i,j] = w_ie if self.is_exc[i] else w_ii

        self.W_out = None
        self.V = np.array([n.v for n in self.neurons])

    def _step(self, u: np.ndarray, spikes_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        I_rec = self.W_rec @ spikes_prev
        I_inp = self.W_in @ u
        I_total = I_rec + I_inp

        spikes = np.zeros(self.n, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            _, spiked = neuron.step(I_total[i])
            spikes[i] = spiked
        return spikes, np.array([n.v for n in self.neurons])

    def run(self, inputs: np.ndarray, washout: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (spike_matrix, state_matrix) after washout."""
        for n in self.neurons:
            n.v = -65.0; n.u = n.p.b * n.v
        spikes_prev = np.zeros(self.n, dtype=bool)
        spike_log = []
        state_log = []
        for t, u in enumerate(inputs):
            spikes, V = self._step(u, spikes_prev)
            spikes_prev = spikes
            if t >= washout:
                spike_log.append(spikes.copy())
                state_log.append(V.copy())
        return np.array(spike_log), np.array(state_log)

    def train_readout(self, inputs: np.ndarray, targets: np.ndarray,
                      washout: int = 50, ridge: float = 1e-4):
        """Train linear readout on spike counts via ridge regression."""
        spikes, _ = self.run(inputs, washout)
        T = targets[washout:]
        # Use spike counts in sliding window as features
        window = 10
        features = np.array([spikes[max(0,i-window):i+1].mean(axis=0)
                              for i in range(len(spikes))])
        self.W_out = np.linalg.solve(
            features.T @ features + ridge * np.eye(self.n),
            features.T @ T
        ).T

    def predict(self, inputs: np.ndarray, washout: int = 50) -> np.ndarray:
        spikes, _ = self.run(inputs, washout)
        window = 10
        features = np.array([spikes[max(0,i-window):i+1].mean(axis=0)
                              for i in range(len(spikes))])
        return features @ self.W_out.T


# ─────────────────────────────────────────────────────────────────────────────
# ORGANOID RESERVOIR — the key OI model
# ─────────────────────────────────────────────────────────────────────────────

class OrganoidReservoir:
    """
    Organoid-specific reservoir model.

    Models the MEA-organoid interface:
      - Neurons: AdEx (burst dynamics, adaptation — most accurate to organoids)
      - Plasticity: R-STDP (reward-modulated, like DishBrain)
      - Input: rate-coded MEA stimulation patterns
      - Output: population firing rate decoded by linear readout

    This is the theoretical model we'd validate against FinalSpark data.
    Key parameters map directly to MEA parameters:
      n_electrodes → input channels on the MEA chip
      n_neurons    → estimated organoid neuron count in recording zone
      stim_scale   → stimulation amplitude (μA)
    """

    def __init__(self,
                 n_electrodes: int = 16,      # typical MEA
                 n_neurons: int = 500,
                 n_outputs: int = 2,
                 connection_prob: float = 0.15,
                 stim_scale: float = 50.0,    # pA
                 dt: float = 0.1,
                 learning_rate: float = 0.001,
                 seed: int = 42):

        self.n_el = n_electrodes
        self.n    = n_neurons
        self.n_out = n_outputs
        self.dt   = dt
        self.lr   = learning_rate
        self.rng  = np.random.default_rng(seed)

        # Neurons: AdEx with heterogeneous parameters (biological variability)
        self.neurons = []
        for _ in range(n_neurons):
            # Add biological noise to parameters
            p = AdExParams(
                C_m=200 + self.rng.normal(0, 20),
                g_L=10  + self.rng.normal(0, 1),
                E_L=-70 + self.rng.normal(0, 5),
                a=2.0   + self.rng.normal(0, 0.5),
                b=100   + self.rng.normal(0, 20),
                tau_w=30 + self.rng.normal(0, 5)
            )
            self.neurons.append(AdExNeuron(p, dt=dt))

        # Stimulation electrode → neuron mapping (spatial proximity)
        self.stim_map = self.rng.uniform(0, stim_scale, (n_neurons, n_electrodes))
        mask = self.rng.random((n_neurons, n_electrodes)) > 0.3  # sparse
        self.stim_map[mask] = 0

        # Recurrent connections (plastic)
        conn_mask = self.rng.random((n_neurons, n_neurons)) < connection_prob
        np.fill_diagonal(conn_mask, False)
        init_W = self.rng.exponential(0.3, (n_neurons, n_neurons))
        self.W_rec = init_W * conn_mask

        # R-STDP learning
        self.stdp = RewardModulatedSTDP(
            n_pre=n_neurons, n_post=n_neurons,
            w_init=self.W_rec,
            params=STDPParams(A_plus=0.005, A_minus=0.005)
        )

        # Readout (trained)
        self.W_out = None
        self.spike_history: List[np.ndarray] = []

    def stimulate(self, pattern: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """
        Apply one stimulation step to the organoid.
        pattern: (n_electrodes,) float — stimulation amplitudes
        reward:  scalar — feedback signal for R-STDP
        Returns: (n_neurons,) spike vector
        """
        I_stim = self.stim_map @ pattern
        I_rec  = self.W_rec @ (self.spike_history[-1] if self.spike_history else
                               np.zeros(self.n))
        I_total = I_stim + I_rec * 100  # scale recurrent

        spikes = np.zeros(self.n, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            _, _, spiked = neuron.step(I_total[i])
            spikes[i] = spiked

        # Update plastic synapses with reward signal
        if reward != 0:
            self.W_rec = self.stdp.step(spikes, spikes, reward, self.dt)

        self.spike_history.append(spikes.astype(float))
        return spikes

    def get_state(self, window: int = 20) -> np.ndarray:
        """Population firing rate over recent window — reservoir state vector."""
        if not self.spike_history:
            return np.zeros(self.n)
        recent = np.array(self.spike_history[-window:])
        return recent.mean(axis=0)

    def train_readout(self, states: np.ndarray, targets: np.ndarray, ridge: float = 1e-4):
        """Fit linear readout W_out from population states to targets."""
        self.W_out = np.linalg.solve(
            states.T @ states + ridge * np.eye(self.n),
            states.T @ targets
        ).T

    def decode(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply trained readout to current or provided state."""
        if self.W_out is None:
            raise RuntimeError("Readout not trained. Call train_readout first.")
        s = state if state is not None else self.get_state()
        return self.W_out @ s

    def reset(self):
        for n in self.neurons:
            n.V = n.p.E_L; n.w = 0.0
        self.spike_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# RESERVOIR QUALITY ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class ReservoirAnalyzer:
    """
    Measures the computational quality of a reservoir.

    Three key properties for organoid computing:
    1. Echo State Property (ESP)       — reservoir forgets initial conditions
    2. Memory Capacity (MC)            — how far back can it remember inputs?
    3. Separation Property (SP)        — can it separate different input patterns?
    4. Kernel Quality (KQ)             — rank of state matrix (expressiveness)

    These metrics answer: is this organoid actually computing anything useful?
    """

    @staticmethod
    def echo_state_property(esn: EchoStateNetwork, input_len: int = 500,
                            n_trials: int = 10) -> float:
        """
        Test ESP by running same input from random initial states.
        Returns convergence ratio (1.0 = perfect ESP, 0.0 = chaos).
        """
        u = np.random.randn(input_len, esn.n_in)
        final_states = []
        for _ in range(n_trials):
            esn.state = np.random.randn(esn.n_res)
            for inp in u:
                esn._step(inp)
            final_states.append(esn.state.copy())
        dists = [np.linalg.norm(final_states[i] - final_states[0])
                 for i in range(1, n_trials)]
        return 1.0 - np.clip(np.mean(dists) / 2.0, 0, 1)

    @staticmethod
    def memory_capacity(esn: EchoStateNetwork, max_delay: int = 50,
                        seq_len: int = 2000) -> Tuple[float, np.ndarray]:
        """
        Compute memory capacity (Jaeger 2002).
        MC = sum of squared correlations between current output and delayed input.
        Theoretical max = n_reservoir.
        Returns: (total_MC, per_delay_correlation²)
        """
        u = np.random.uniform(-1, 1, seq_len + max_delay)
        inputs = u[max_delay:].reshape(-1, 1)
        states = esn.run(inputs, washout=100)
        mc_per_delay = np.zeros(max_delay)
        for k in range(1, max_delay + 1):
            target = u[max_delay - k: seq_len + max_delay - k - 100]
            y = states @ np.linalg.lstsq(states, target, rcond=None)[0]
            corr = np.corrcoef(y, target)[0, 1]
            mc_per_delay[k-1] = corr**2
        return float(np.sum(mc_per_delay)), mc_per_delay

    @staticmethod
    def separation_property(lsm: LiquidStateMachine,
                            n_pairs: int = 20, T: int = 200) -> float:
        """
        Measure if reservoir separates different inputs into different states.
        High separation = more computational power.
        Returns mean pairwise state distance (normalized).
        """
        rng = np.random.default_rng(0)
        states = []
        for _ in range(n_pairs):
            u = rng.integers(0, 2, (T, lsm.n_in)).astype(float)
            spikes, _ = lsm.run(u, washout=20)
            states.append(spikes.mean(axis=0))
        states = np.array(states)
        dists = []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                dists.append(np.linalg.norm(states[i] - states[j]))
        return float(np.mean(dists))

    @staticmethod
    def kernel_quality(esn: EchoStateNetwork, input_len: int = 500) -> int:
        """
        Kernel quality = rank of reservoir state matrix.
        Higher rank = more linearly independent dimensions = more expressiveness.
        """
        u = np.random.randn(input_len, esn.n_in)
        states = esn.run(u, washout=50)
        return int(np.linalg.matrix_rank(states))


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RESERVOIR COMPUTING — OI FOUNDATION")
    print("=" * 60)

    # --- ESN: learn NARMA-10 task (standard benchmark) ---
    print("\n[1] Echo State Network — NARMA-10 benchmark")
    rng = np.random.default_rng(0)
    T = 3000
    u = rng.uniform(0, 0.5, T).reshape(-1, 1)

    # NARMA-10: nonlinear autoregressive moving average (order 10)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3*y[t-1] + 0.05*y[t-1]*sum(y[t-k] for k in range(1,11))
                + 1.5*u[t-1,0]*u[t-10,0] + 0.1)

    esn = EchoStateNetwork(n_inputs=1, n_reservoir=200, n_outputs=1,
                           spectral_radius=0.95, leak_rate=0.3)
    split = 2000
    esn.train(u[:split], y[:split, None], washout=100)
    pred = esn.predict(u[split:], washout=100).flatten()
    targ = y[split+100:]
    nmse = np.mean((pred - targ)**2) / np.var(targ)
    print(f"   NARMA-10 NMSE: {nmse:.4f} (< 0.1 is good)")

    # --- Reservoir quality metrics ---
    print("\n[2] Reservoir Quality Metrics")
    analyzer = ReservoirAnalyzer()
    esp  = analyzer.echo_state_property(esn)
    mc, mc_delays = analyzer.memory_capacity(esn, max_delay=30)
    kq   = analyzer.kernel_quality(esn)
    print(f"   Echo State Property: {esp:.3f}")
    print(f"   Memory Capacity:     {mc:.1f} (max={esn.n_res})")
    print(f"   Kernel Quality:      {kq} / {esn.n_res}")

    # --- OrganoidReservoir: basic run ---
    print("\n[3] Organoid Reservoir — MEA stimulation simulation")
    organoid = OrganoidReservoir(n_electrodes=16, n_neurons=200, dt=0.5)
    states = []
    for t in range(500):
        pattern = (rng.random(16) > 0.7).astype(float)
        reward  = 1.0 if t % 20 == 0 else 0.0
        spikes  = organoid.stimulate(pattern, reward=reward)
        states.append(organoid.get_state(window=10))
    mean_firing = np.mean([s.mean() for s in states])
    print(f"   Mean firing rate: {mean_firing*1000:.1f} Hz (target: 5-30 Hz for healthy organoid)")
    print(f"   Active neurons:   {int(np.mean([s > 0.01 for s in states])*organoid.n)}/{organoid.n}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reservoir Computing Foundation — Organoid Intelligence", fontsize=13)

    axes[0,0].plot(targ[:300], label='Target', color='steelblue', lw=1)
    axes[0,0].plot(pred[:300], label='ESN', color='coral', lw=1, ls='--')
    axes[0,0].set_title(f"ESN NARMA-10 (NMSE={nmse:.4f})")
    axes[0,0].legend()

    axes[0,1].bar(range(len(mc_delays)), mc_delays, color='mediumseagreen')
    axes[0,1].set_title(f"Memory Capacity per Delay (Total={mc:.1f})")
    axes[0,1].set_xlabel("Delay (steps)")

    state_matrix = np.array(states)
    im = axes[1,0].imshow(state_matrix.T, aspect='auto', cmap='hot',
                           interpolation='none')
    axes[1,0].set_title("Organoid State Matrix (neurons × time)")
    axes[1,0].set_xlabel("Time step"); axes[1,0].set_ylabel("Neuron index")
    plt.colorbar(im, ax=axes[1,0])

    firing_rates = state_matrix.mean(axis=1) * 1000
    axes[1,1].plot(firing_rates, color='steelblue', lw=0.8)
    axes[1,1].set_title("Population Firing Rate Over Time")
    axes[1,1].set_xlabel("Time step"); axes[1,1].set_ylabel("Rate (Hz)")
    axes[1,1].axhline(30, color='red', ls='--', alpha=0.5, label='30 Hz ceiling')
    axes[1,1].axhline(5,  color='orange', ls='--', alpha=0.5, label='5 Hz floor')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig("experiments/results/reservoir_foundation.png", dpi=150)
    print("\nSaved: experiments/results/reservoir_foundation.png")
    plt.show()
