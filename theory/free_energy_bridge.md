# Free Energy Principle â†” Organoid Intelligence: Formal Derivation
*Tyson Guerrero + Claude, May 2026 â€” working draft*

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

The organoid has an implicit **generative model** of its world â€” the MEA stimulation
it expects to receive given its current activity. We formalize this as:

```
p(o, x, W) = p(o | x) Â· p(x | W) Â· p(W)
```

**Variables:**
| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| `o` | MEA observation vector (stimulation + recorded spikes) | â„^E  (E = electrodes) |
| `x` | Network activity vector (firing rates at time t) | â„^N  (N = neurons) |
| `W` | Synaptic weight matrix | â„^{NÃ—N} |

**Likelihood** `p(o | x)`: observations given network state.
Assume Gaussian noise on recorded spikes:
```
p(o | x) = N(CÂ·x, Î£_o)
```
where C is the recording matrix (MEA electrode sensitivity, fixed).

**Dynamics prior** `p(x | W)`: what activity does the network expect?
The network dynamics are:
```
Ï„ áº‹ = -x + Ïƒ(WÂ·x + I_stim)
```
At approximate fixed point x*, the prior over x is:
```
p(x | W) â‰ˆ N(x*, Î£_x)    where x* = Ïƒ(WÂ·x* + I_stim)
```

**Weight prior** `p(W)`: regularization. Assume log-normal (spike weights
are positive, sparse):
```
ln p(W) = -Î»/2 Â· ||W||_FÂ²  +  const
```

---

## Step 2: Variational Free Energy

We want to compute F. But p(x | o) is intractable (nonlinear dynamics).
Use a **mean-field variational approximation** q(x, W) = q(x) Â· q(W).

Free energy:
```
F = E_q[ln q(x,W) - ln p(o,x,W)]
  = E_q[ln q(x) - ln p(o|x)] + E_q[ln q(W) - ln p(W)] + const
  = F_perceptual + F_weights
```

### F_perceptual (perception term)
```
F_perceptual = E_q[ln q(x)] - E_q[ln p(o|x)]
             = -H[q(x)] + (1/2)(CÂ·Î¼_x - o)^T Î£_o^{-1} (CÂ·Î¼_x - o)
             = -H[q(x)] + (1/2)||Îµ||Â²_Î£o
```
where `Îµ = CÂ·Î¼_x - o` is the **prediction error** â€” the difference between
what the network predicts it should see and what it actually sees.

### F_weights (weight term)
```
F_weights = E_q[ln q(W)] - E_q[ln p(W)]
          = -H[q(W)] + (Î»/2) E_q[||W||_FÂ²]
```

---

## Step 3: Gradient with Respect to Weights

We want `âˆ‚F/âˆ‚W_ij`:

```
âˆ‚F/âˆ‚W_ij = âˆ‚F_perceptual/âˆ‚W_ij + âˆ‚F_weights/âˆ‚W_ij
```

### Term 1: Perceptual gradient
```
âˆ‚F_perceptual/âˆ‚W_ij = âˆ‚/âˆ‚W_ij [(1/2)||CÂ·x - o||Â²_Î£o]
                     = (C^T Î£_o^{-1} (CÂ·x - o))^T Â· âˆ‚x/âˆ‚W_ij
                     = Î´^T Â· âˆ‚x/âˆ‚W_ij
```
where `Î´ = C^T Î£_o^{-1} (CÂ·x - o)` = **precision-weighted prediction error**.

Now we need `âˆ‚x/âˆ‚W_ij`. From the network dynamics at steady state:
```
x_i = Ïƒ(Î£_k W_ik x_k + I_i)
âˆ‚x_i/âˆ‚W_ij = Ïƒ'(h_i) Â· x_j   where h_i = Î£_k W_ik x_k + I_i
```

Therefore:
```
âˆ‚F_perceptual/âˆ‚W_ij = Î´_i Â· Ïƒ'(h_i) Â· x_j
```

### Term 2: Weight regularization gradient
```
âˆ‚F_weights/âˆ‚W_ij = Î» Â· W_ij
```

### Combined gradient
```
âˆ‚F/âˆ‚W_ij = Î´_i Â· Ïƒ'(h_i) Â· x_j + Î» Â· W_ij
```

**Weight update** (gradient descent, step size Î·):
```
Î”W_ij = -Î· Â· âˆ‚F/âˆ‚W_ij
       = -Î· Â· Î´_i Â· Ïƒ'(h_i) Â· x_j  -  Î·Â·Î» Â· W_ij
         â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
              STDP-like term          weight decay
```

---

## Step 4: The STDP Equivalence

### R-STDP update rule:
```
Î”W_ij = r(t) Â· e_ij(t)
```
where:
- `r(t)` = reward/feedback signal
- `e_ij(t)` = STDP eligibility trace = âˆ« A_+ Â· z_j(s) Â· Î´(t_post - s) ds

### Identifying the terms:

**The reward signal `r(t)` = precision-weighted prediction error `Î´_i`:**
```
r(t) â‰¡ Î´_i = (C^T Î£_o^{-1})(CÂ·x - o)_i
```

When ball HIT â†’ o matches prediction â†’ Î´ small â†’ weak update
When ball MISS â†’ o â‰  prediction â†’ Î´ large â†’ strong update

But DishBrain flips this: **chaos** on miss, **ordered** on hit.
- Chaos = high-entropy stimulation â†’ increases |Îµ| â†’ large |Î´| â†’ large update
- Ordered = low-entropy stimulation â†’ decreases |Îµ| â†’ small |Î´| â†’ small update

This is EXACTLY precision modulation of prediction error. âœ“

**The eligibility trace `e_ij(t)` = `Ïƒ'(h_i) Â· x_j`:**

Pre-synaptic activity `x_j` is a low-pass filtered spike train â€” matches
the STDP pre-trace `z_j(t)`.

Post-synaptic term `Ïƒ'(h_i)` â‰ˆ spiking probability derivative â€” matches
the post-spike STDP kernel when using a smooth threshold function.

Therefore:
```
R-STDP = âˆ‚F/âˆ‚W  (up to proportionality constant Î·)
         WHEN r(t) encodes prediction error
              e_ij(t) encodes activity correlation
```

---

## Step 5: Predictions

This derivation generates **3 testable predictions** our simulation can check:

### P1: Learning rate âˆ precision (Î£_o^{-1})
Higher SNR on MEA â†’ higher precision â†’ faster learning.
*In simulation: add Gaussian noise to MEA readout, verify slower learning.*

### P2: Spontaneous activity â‰ˆ prior mean x*
Between stimulation trials, neurons should settle toward the activity
pattern that minimizes F under the prior â€” which is the most recently
learned pattern.
*In simulation: record activity during "off" periods, compare to recent patterns.*

### P3: Optimal feedback = maximal precision-weighted prediction error
The best stimulation protocol is the one that maximizes |Î´| when
behavior is wrong and minimizes it when correct.
- More contrast between chaos/ordered â†’ faster learning
- This gives us a principled design rule for MEA stimulation protocols

---

## Open Issues

### Issue 1: Mean-field validity
We assumed q(x, W) = q(x) Â· q(W). This ignores correlations between
weights and activity. For spiking networks this is a real approximation.
Future work: use structured variational family (full-covariance q(x)).

### Issue 2: Discrete-time formulation
The above is continuous-time FEP. STDP rules are traditionally stated
in discrete spike-time differences. Need to formalize the mapping
between continuous F-gradient and discrete STDP window.
*Reference: Da Costa et al. 2020 â€” "Active inference on discrete state spaces"*

### Issue 3: Stochastic spikes
Ïƒ'(h_i) is the smooth approximation to the spike threshold derivative.
Real neurons spike stochastically. The above holds for the mean-field
approximation but breaks down at the single-cell level.
*This is OK: organoids use population-level readout, mean-field is appropriate.*

### Issue 4: The sign of chaos stimulation
The DishBrain convention: chaos = punishment. In FEP terms, chaos
INCREASES prediction error Îµ. We need to verify that increasing Îµ
on negative reward actually drives the weights in the correct direction.
*This requires careful analysis of which direction Î´_i points.*

---

## Status

- [x] Generative model defined
- [x] Free energy decomposed
- [x] Gradient âˆ‚F/âˆ‚W derived
- [x] STDP equivalence identified
- [x] 3 testable predictions derived
- [ ] Issue 2: discrete-time formulation (next session)
- [ ] Issue 4: sign analysis (next session)
- [ ] Simulation: test P1 (noise â†’ learning rate)
- [ ] Write as 4-page theory note for arXiv

---

## Key Insight (plain English)

The organoid is minimizing surprise.

The chaos stimulation on a miss is literally a high-surprise event â€”
lots of unpredictable signal coming in. The brain hates unpredictability
(this is the whole Free Energy Principle in one sentence). So the organoid
changes its weights to avoid the situations that produce chaos.

Ordered stimulation on a hit is low-surprise â€” expected, regular.
Neurons don't need to change much.

The DishBrain researchers stumbled onto this protocol empirically in 2022.
The FEP tells us *why it works* from first principles â€” and lets us
design better protocols by maximizing the surprise contrast.

This is the actual contribution of the theoretical work.
