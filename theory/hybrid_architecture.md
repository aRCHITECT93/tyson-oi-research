# Hybrid Biological-Silicon Architecture (HBSA)
*Theoretical Framework — Tyson, May 2026*

---

## The Core Idea

Nobody has formally specified how to combine GPU compute with
organoid compute into a single working system. This document
builds that specification from first principles.

---

## 1. Why Hybrid?

### What organoids can do that silicon can't (efficiently):

**1. Energy efficiency at scale**
- Human neuron: ~10 fJ per spike
- GPU equivalent op: ~1 pJ (100x more expensive)
- At 1M neurons × 10 Hz: organoid = 100 mW, GPU = 10 W

**2. Temporal integration with adaptation**
- Spike-frequency adaptation + STDP = natural working memory
- GPUs need explicit memory allocations + attention mechanisms
- Organoid memory is substrate-native

**3. Distribution shift robustness**
- Organoids have shown resilience to novel stimulation patterns
- Silicon models degrade sharply outside training distribution
- Biological noise acts as natural regularization

**4. Low-dimensional pattern completion**
- 500 organoid neurons can complete a pattern from partial input
- Equivalent Hopfield network on GPU needs explicit architecture

### What silicon can do that organoids can't (yet):

**1. Exact arithmetic**
- Organoids are noisy analog computers
- Financial calculations, cryptography, databases: silicon only

**2. High-speed parallel throughput**
- RTX 5070 Ti: 43 TFLOPS FP16
- 1M organoid neurons @ 10 Hz: ~10 MOPS equivalent
- Speed gap: ~4 million x in favor of silicon (for parallelizable tasks)

**3. Reproducibility**
- Same weights → same output (silicon)
- Organoids drift, die, vary between cultures

**4. Programmability**
- Silicon runs any algorithm you write
- Organoids: you set stimulation, they decide what to do

---

## 2. The HBSA Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TASK ROUTER                          │
│         (GPU-based, learned routing policy)             │
└─────────┬───────────────────────┬───────────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────┐    ┌─────────────────────────────┐
│   SILICON LAYER │    │     BIOLOGICAL LAYER        │
│                 │    │                             │
│  Exact compute  │    │  Organoid reservoir(s)      │
│  LLM inference  │◄──►│  MEA interface              │
│  Vector search  │    │  Temporal integration       │
│  Optimization   │    │  Adaptive pattern matching  │
└─────────────────┘    └─────────────────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌───────────────┐
              │  INTEGRATION  │
              │    LAYER      │
              │  (GPU-based)  │
              └───────────────┘
```

### 2.1 Task Router
A learned policy (trainable via RL) that:
- Receives task description (as embedding)
- Routes to silicon, biological, or hybrid pipeline
- Learned routing objective: minimize latency + energy + error

### 2.2 Biological Layer Interface
```
Silicon → Biological:
  1. Encode input as stimulation pattern (MEAEncoder)
  2. Apply via FinalSpark API (or local MEA device)
  3. Wait for response window (5–50ms)
  
Biological → Silicon:
  1. Record spike patterns from MEA
  2. Decode via trained linear readout
  3. Return as floating point tensor
```

### 2.3 Latency Model
```
T_biological = T_stim + T_propagation + T_readout
             ≈ 1ms + 20ms + 5ms
             = ~26ms per query

T_silicon    = depends on model
             = ~1ms for small networks
             = ~100ms for LLM inference
```

**Key insight:** For tasks that take >26ms on silicon AND where
biological compute quality is acceptable — route to organoid.

---

## 3. Routing Decision Theory

### 3.1 Task Classification Features
```python
features = {
    "uncertainty":    float,   # how uncertain is the task?
    "temporal_depth": int,     # how many time steps of memory needed?
    "parallelism":    float,   # can it be parallelized?
    "exact_required": bool,    # does answer need to be exact?
    "latency_budget": float,   # milliseconds available
}
```

### 3.2 Routing Rule (heuristic v1)
```python
def route(features):
    if features["exact_required"]:
        return "silicon"
    if features["temporal_depth"] > 100 and features["uncertainty"] > 0.7:
        return "biological"
    if features["parallelism"] > 0.8:
        return "silicon"
    if features["latency_budget"] < 30:
        return "silicon"
    return "hybrid"
```

### 3.3 Learned Routing (v2)
Train a small NN on (task_features, routed_to, outcome) triples.
Outcome = (latency, energy, accuracy).
Optimize for Pareto frontier.

---

## 4. Information-Theoretic Analysis

### 4.1 Channel Capacity of the MEA Interface

The MEA is a noisy channel between silicon and organoid.
By Shannon's theorem:
```
C = B · log₂(1 + SNR)
```

For a 64-electrode MEA:
- B ≈ 10 kHz per channel
- SNR ≈ 20 dB (typical MEA recording)
- C ≈ 64 × 10k × log₂(101) ≈ 4.2 Mbps

**Implication:** The biological layer can receive at most ~4 Mbps
of useful information from silicon. This bottleneck constrains
what tasks are feasible to route to the organoid.

### 4.2 Effective Organoid Bandwidth
With N neurons, firing rate f, and readout resolution R:
```
B_organoid ≈ N × f × R
           = 500k × 10 Hz × 1 bit
           = 5 Mbps outgoing
```

Roughly matched to MEA input bandwidth — the system is balanced.

### 4.3 Energy Efficiency Crossover
At what task complexity does biological outperform silicon?

```
E_silicon   = FLOPS × energy_per_FLOP
E_biological = N_spikes × energy_per_spike

Break-even: FLOPS > (E_bio / E_silicon_per_FLOP) × N_spikes
                  ≈ (100mJ / 1pJ) × 10,000
                  ≈ 10^15 FLOPS
```

Above ~10^15 FLOPS of silicon-equivalent work per decision:
biological compute is more energy-efficient.
This is in reach for complex temporal reasoning tasks.

---

## 5. Simulation Plan

To test HBSA before we have a real organoid:

1. **Simulate organoid layer** → OrganoidReservoir (done)
2. **Add silicon layer** → PyTorch GPU model (done)
3. **Implement router** → simple heuristic first
4. **Benchmark on tasks:**
   - Chaotic time series prediction (organoid advantage)
   - MNIST classification (silicon advantage)
   - NARMA-30 (mixed)
   - Pong with distribution shift (test robustness claim)

5. **Measure:**
   - Accuracy by layer
   - Energy (proxy: FLOPs vs spike count)
   - Latency
   - Routing decision quality

---

## 6. Path to Real Implementation

### Near-term (simulation):
- Build full HBSA in software with OrganoidReservoir
- Characterize routing decision quality

### Medium-term (FinalSpark API):
- Replace simulated organoid with real FinalSpark neuroplatform
- Validate that simulation predictions hold

### Long-term (custom MEA):
- Cortical Labs sells DIY MEA kits
- Could run organoid experiments locally with university lab access
- Cost: ~$50k for basic setup (realistically needs lab partnership)

---

## Status
- [ ] Implement routing heuristic in code
- [ ] Run benchmark suite on simulated HBSA
- [ ] Write up information-theoretic analysis more rigorously
- [ ] Find collaborators with MEA access
