# Free Energy Principle ↔ Organoid Intelligence: Formal Derivation
*Tyson + Claude, May 2026 — working draft*

---

## The Claim

**R-STDP is gradient descent on variational free energy.**

If this is true, it gives us:
1. A *principled* explanation for why organoids learn (not just "they do")
2. Novel predictions about learning speed and optimal stimulation
3. A bridge between AI theory (variational inference) and wet neuroscience
4. A publishable theoretical result independent of simulation results

---

## Step 1: Define the Generative Model

The organoid has an implicit **generative model** of its world — the MEA stimulation
it expects to receive given its current activity. We formalize this as:

```
p(o, x, W) = p(o | x) · p(x | W) · p(W)
```

**Variables:**
| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| `o` | MEA observation vector (stimulation + recorded spikes) | ℝ^E  (E = electrodes) |
| `x` | Network activity vector (firing rates at time t) | ℝ^N  (N = neurons) |
| `W` | Synaptic weight matrix | ℝ^{N×N} |

**Likelihood** `p(o | x)`: observations given network state.
Assume Gaussian noise on recorded spikes:
```
p(o | x) = N(C·x, Σ_o)
```
where C is the recording matrix (MEA electrode sensitivity, fixed).

**Dynamics prior** `p(x | W)`: what activity does the network expect?
The network dynamics are:
```
τ ẋ = -x + σ(W·x + I_stim)
```
At approximate fixed point x*, the prior over x is:
```
p(x | W) ≈ N(x*, Σ_x)    where x* = σ(W·x* + I_stim)
```

**Weight prior** `p(W)`: regularization. Assume log-normal (spike weights
are positive, sparse):
```
ln p(W) = -λ/2 · ||W||_F²  +  const
```

---

## Step 2: Variational Free Energy

We want to compute F. But p(x | o) is intractable (nonlinear dynamics).
Use a **mean-field variational approximation** q(x, W) = q(x) · q(W).

Free energy:
```
F = E_q[ln q(x,W) - ln p(o,x,W)]
  = E_q[ln q(x) - ln p(o|x)] + E_q[ln q(W) - ln p(W)] + const
  = F_perceptual + F_weights
```

### F_perceptual (perception term)
```
F_perceptual = E_q[ln q(x)] - E_q[ln p(o|x)]
             = -H[q(x)] + (1/2)(C·μ_x - o)^T Σ_o^{-1} (C·μ_x - o)
             = -H[q(x)] + (1/2)||ε||²_Σo
```
where `ε = C·μ_x - o` is the **prediction error** — the difference between
what the network predicts it should see and what it actually sees.

### F_weights (weight term)
```
F_weights = E_q[ln q(W)] - E_q[ln p(W)]
          = -H[q(W)] + (λ/2) E_q[||W||_F²]
```

---

## Step 3: Gradient with Respect to Weights

We want `∂F/∂W_ij`:

```
∂F/∂W_ij = ∂F_perceptual/∂W_ij + ∂F_weights/∂W_ij
```

### Term 1: Perceptual gradient
```
∂F_perceptual/∂W_ij = ∂/∂W_ij [(1/2)||C·x - o||²_Σo]
                     = (C^T Σ_o^{-1} (C·x - o))^T · ∂x/∂W_ij
                     = δ^T · ∂x/∂W_ij
```
where `δ = C^T Σ_o^{-1} (C·x - o)` = **precision-weighted prediction error**.

Now we need `∂x/∂W_ij`. From the network dynamics at steady state:
```
x_i = σ(Σ_k W_ik x_k + I_i)
∂x_i/∂W_ij = σ'(h_i) · x_j   where h_i = Σ_k W_ik x_k + I_i
```

Therefore:
```
∂F_perceptual/∂W_ij = δ_i · σ'(h_i) · x_j
```

### Term 2: Weight regularization gradient
```
∂F_weights/∂W_ij = λ · W_ij
```

### Combined gradient
```
∂F/∂W_ij = δ_i · σ'(h_i) · x_j + λ · W_ij
```

**Weight update** (gradient descent, step size η):
```
ΔW_ij = -η · ∂F/∂W_ij
       = -η · δ_i · σ'(h_i) · x_j  -  η·λ · W_ij
         ─────────────────────────    ───────────
              STDP-like term          weight decay
```

---

## Step 4: The STDP Equivalence

### R-STDP update rule:
```
ΔW_ij = r(t) · e_ij(t)
```
where:
- `r(t)` = reward/feedback signal
- `e_ij(t)` = STDP eligibility trace = ∫ A_+ · z_j(s) · δ(t_post - s) ds

### Identifying the terms:

**The reward signal `r(t)` = precision-weighted prediction error `δ_i`:**
```
r(t) ≡ δ_i = (C^T Σ_o^{-1})(C·x - o)_i
```

When ball HIT → o matches prediction → δ small → weak update
When ball MISS → o ≠ prediction → δ large → strong update

But DishBrain flips this: **chaos** on miss, **ordered** on hit.
- Chaos = high-entropy stimulation → increases |ε| → large |δ| → large update
- Ordered = low-entropy stimulation → decreases |ε| → small |δ| → small update

This is EXACTLY precision modulation of prediction error. ✓

**The eligibility trace `e_ij(t)` = `σ'(h_i) · x_j`:**

Pre-synaptic activity `x_j` is a low-pass filtered spike train — matches
the STDP pre-trace `z_j(t)`.

Post-synaptic term `σ'(h_i)` ≈ spiking probability derivative — matches
the post-spike STDP kernel when using a smooth threshold function.

Therefore:
```
R-STDP = ∂F/∂W  (up to proportionality constant η)
         WHEN r(t) encodes prediction error
              e_ij(t) encodes activity correlation
```

---

## Step 5: Predictions

This derivation generates **3 testable predictions** our simulation can check:

### P1: Learning rate ∝ precision (Σ_o^{-1})
Higher SNR on MEA → higher precision → faster learning.
*In simulation: add Gaussian noise to MEA readout, verify slower learning.*

### P2: Spontaneous activity ≈ prior mean x*
Between stimulation trials, neurons should settle toward the activity
pattern that minimizes F under the prior — which is the most recently
learned pattern.
*In simulation: record activity during "off" periods, compare to recent patterns.*

### P3: Optimal feedback = maximal precision-weighted prediction error
The best stimulation protocol is the one that maximizes |δ| when
behavior is wrong and minimizes it when correct.
- More contrast between chaos/ordered → faster learning
- This gives us a principled design rule for MEA stimulation protocols

---

## Open Issues

### Issue 1: Mean-field validity
We assumed q(x, W) = q(x) · q(W). This ignores correlations between
weights and activity. For spiking networks this is a real approximation.
Future work: use structured variational family (full-covariance q(x)).

### Issue 2: Discrete-time formulation
The above is continuous-time FEP. STDP rules are traditionally stated
in discrete spike-time differences. Need to formalize the mapping
between continuous F-gradient and discrete STDP window.
*Reference: Da Costa et al. 2020 — "Active inference on discrete state spaces"*

### Issue 3: Stochastic spikes
σ'(h_i) is the smooth approximation to the spike threshold derivative.
Real neurons spike stochastically. The above holds for the mean-field
approximation but breaks down at the single-cell level.
*This is OK: organoids use population-level readout, mean-field is appropriate.*

### Issue 4: The sign of chaos stimulation
The DishBrain convention: chaos = punishment. In FEP terms, chaos
INCREASES prediction error ε. We need to verify that increasing ε
on negative reward actually drives the weights in the correct direction.
*This requires careful analysis of which direction δ_i points.*

---

## Status

- [x] Generative model defined
- [x] Free energy decomposed
- [x] Gradient ∂F/∂W derived
- [x] STDP equivalence identified
- [x] 3 testable predictions derived
- [ ] Issue 2: discrete-time formulation (next session)
- [ ] Issue 4: sign analysis (next session)
- [ ] Simulation: test P1 (noise → learning rate)
- [ ] Write as 4-page theory note for arXiv

---

## Key Insight (plain English)

The organoid is minimizing surprise.

The chaos stimulation on a miss is literally a high-surprise event —
lots of unpredictable signal coming in. The brain hates unpredictability
(this is the whole Free Energy Principle in one sentence). So the organoid
changes its weights to avoid the situations that produce chaos.

Ordered stimulation on a hit is low-surprise — expected, regular.
Neurons don't need to change much.

The DishBrain researchers stumbled onto this protocol empirically in 2022.
The FEP tells us *why it works* from first principles — and lets us
design better protocols by maximizing the surprise contrast.

This is the actual contribution of the theoretical work.
