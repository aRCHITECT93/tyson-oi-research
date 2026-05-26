# Open Problems in Organoid Intelligence
*Living document — updated as research progresses*
*Author: Tyson | Last updated: May 2026*

---

## Tier 1 — Foundational Gaps (Publishable today)

### 1. The Encoding Problem
**What:** How do you optimally map information INTO an organoid via MEA stimulation?

Current state:
- DishBrain used simple frequency-modulated pulses
- FinalSpark uses rate coding
- Nobody has formally optimized the encoding scheme

**The gap:** There is NO principled theory for optimal input encoding given
organoid topology, connectivity, and the specific learning rule (R-STDP).

**Our angle:**
- Frame it as a rate-distortion theory problem
- Model organoid as a noisy channel (Shannon)
- Find the encoding that maximizes mutual information between stimulation and
  stable reservoir state
- Compare: rate coding vs temporal coding vs population coding vs burst coding

**Key equation to formalize:**
```
I*(S; X) = max_{encoding} I(S; f(X))
subject to: metabolic cost C(S) ≤ C_max
            organoid stability constraint: SR(W_rec) < 1
```

**References needed:**
- Rate-distortion theory (Cover & Thomas)
- Neural coding theory (Dayan & Abbott, ch. 1-3)
- MEA stimulation literature (Wagenaar et al.)

---

### 2. Organoid Scaling Laws
**What:** Do OI systems follow power-law scaling like LLMs?

Current state:
- LLMs: loss ∝ N^(-α) where N = parameters (Chinchilla laws)
- SNNs: some evidence of similar laws (not well characterized)
- Organoids: NO systematic study exists

**The gap:** No one has mapped how organoid performance scales with:
- Neuron count (N)
- MEA electrode density (E)
- Training time / stimulus duration (T)
- Organoid maturation (age in days)

**Our angle:**
- Use simulation (our OrganoidReservoir) to generate synthetic scaling data
- Fit power laws to memory capacity, separation property as function of N
- Extrapolate to real organoid sizes (1M–1B neurons)
- Compare to silicon equivalents on same tasks

**Hypothesis:** Organoids have fundamentally different scaling exponents than
silicon due to:
  a) Spike sparsity (only ~5% active at any time)
  b) Metabolic constraints (unlike FLOP constraints in silicon)
  c) Physical 3D connectivity (not planar like GPU matrices)

---

### 3. The Readout Problem
**What:** How do you reliably decode the organoid's "answer" from noisy MEA recordings?

Current state:
- Population firing rate (crude average)
- Linear discriminant analysis on spike trains
- No principled framework

**The gap:** There's no theory connecting:
- The computational state of the organoid (latent)
- The observable MEA spikes (noisy, partial observation)
- The "correct" decoding function

**Our angle:**
- Frame as a hidden Markov model / state-space model
- Organoid state = latent variable
- MEA recording = noisy observation
- Build a Kalman filter / particle filter for state estimation
- Optimal readout = minimum mean-squared error estimator

This is tractable with existing signal processing theory.
Nobody has applied state-space estimation formally to organoid decoding.

---

## Tier 2 — Theoretical Contributions (3–6 months)

### 4. Free Energy Principle Bridge
**What:** Karl Friston's Free Energy Principle (FEP) may be the correct
theoretical framework for why organoids learn at all.

Current state:
- FEP: biological systems minimize variational free energy F = E_q[log q(s) - log p(s,o)]
- This explains perception, action, learning in biological systems
- DishBrain behavior is consistent with FEP (neurons minimize prediction error)
- But nobody has formally mapped FEP to the OI/MEA setting

**The gap:** A formal derivation showing:
  organoid + MEA feedback ≡ active inference under FEP

**Key claim to prove:**
The R-STDP learning rule emerges from minimizing variational free energy
in the organoid's generative model of its stimulus environment.

If true: this gives us a principled theory of WHY organoids learn
(not just empirical observation that they do).

**Mathematical program:**
1. Define organoid's generative model p(o, s | π) 
   - o = MEA observations
   - s = internal states  
   - π = "policies" (firing patterns)
2. Show R-STDP weight updates = gradient of -F
3. Show game feedback = precision-weighted prediction error
4. Derive bounds on learning speed from FEP parameters

---

### 5. Hybrid Silicon-Biological Architecture
**What:** What's the optimal division of labor between organoid compute and GPU compute?

Current state:
- Completely separate research communities
- No formal framework for hybrid computation
- Ad-hoc combinations in a few papers

**The gap:** No one has formalized:
- Which task types are organoids better at? (and provably so)
- Which are GPUs better at?
- How to interface them efficiently (latency, encoding)?

**Theoretical result we want:**
A complexity-theoretic characterization of the organoid computational class.

**Conjecture:**
Organoids are efficient at:
- Tasks requiring temporal integration (memory of sequences)
- Low-energy pattern completion
- Adaptation under distribution shift (non-stationary inputs)

GPUs are efficient at:
- High-parallelism matrix operations
- Exact numerical computation
- Large-scale optimization

A hybrid architecture routes tasks by type → significant efficiency gains.

---

### 6. Consciousness / Sentience Metrics
**What:** Can we define a formal metric for "at what point does an organoid matter morally"?

This is not just philosophy — it has direct experimental implications.
If we have a metric, we can:
- Draw an ethical line in organoid size/complexity
- Design experiments that stay on the right side
- Contribute to policy (EU AI Act has nothing on organoids yet)

**Existing frameworks:**
- Integrated Information Theory (IIT): Φ measure
- Global Workspace Theory: information broadcast
- Higher-Order Theory: representations of representations

**Our angle:**
- Compute Φ (or cheaper approximations like ΦID) on our simulated organoids
- Map how Φ scales with N, connectivity, maturation
- Identify if there's a phase transition (threshold effect)
- Argue for Φ-based ethical threshold in policy context

---

## Tier 3 — Long-term Moonshots (1–2 years)

### 7. Organoid as Foundation Model
Can a sufficiently large, trained organoid perform few-shot learning like an LLM?
What would "pre-training" an organoid even mean biologically?

### 8. Multi-Organoid Networks
FinalSpark and others are moving toward connecting multiple organoids.
What are the information-theoretic constraints on multi-organoid communication?
Is biological "internet" possible?

### 9. Drug/Stimulation Protocol Optimization
Given our reservoir model, can we design optimal stimulation protocols
that maximize learning rate while minimizing biological stress?
This has direct clinical applications (brain-computer interfaces, neurorehabilitation).

---

## Resources & Key Papers

| Paper | DOI/Link | Priority |
|-------|----------|----------|
| DishBrain (Kagan 2022) | 10.1016/j.neuron.2022.09.001 | READ FIRST |
| OI Roadmap (Smirnova 2023) | 10.3389/fsci.2023.1017235 | READ FIRST |
| Free Energy Principle (Friston 2010) | 10.1038/nrn2787 | Core theory |
| Liquid State Machines (Maass 2002) | 10.1162/089976602760407955 | Core theory |
| Echo State Networks (Jaeger 2001) | Available on ScholarlyCommons | Core theory |
| Triplet STDP (Pfister 2006) | 10.1523/JNEUROSCI.1425-06.2006 | STDP learning |
| Integrated Information Theory (Tononi 2004) | 10.1186/1471-2202-5-42 | Consciousness |
| FinalSpark Neuroplatform (2024) | bioRxiv | Current tech |

---

## Experiment Queue

- [ ] Run NARMA benchmark on OrganoidReservoir vs ESN
- [ ] Map memory capacity as function of n_neurons (scaling)
- [ ] Implement Φ calculation on small simulated networks
- [ ] Formalize FEP-STDP equivalence proof (draft)
- [ ] Apply for FinalSpark research access
- [ ] Download Allen Brain Atlas organoid recordings
