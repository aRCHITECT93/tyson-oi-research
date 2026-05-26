# Organoid Intelligence Research Foundation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

*Tyson Guerrero ([@aRCHITECT93](https://github.com/aRCHITECT93)) â€” May 2026*

> Building the theoretical and computational foundation for organoid intelligence (OI) research.
> No wetlab required â€” pure brain power + simulation + theory.

---

## What This Is

This project is a personal research foundation for the field of **organoid intelligence** â€”
the science of using human brain organoids (lab-grown neuron clusters) as computing systems.

The goal is to:
1. **Understand** the computational principles deeply (not just surface-level)
2. **Simulate** organoid behavior before touching real wetware
3. **Develop original theory** around open problems in the field
4. **Publish** findings that contribute to the scientific literature
5. **Eventually interface** with real organoids via FinalSpark API

---

## Project Structure

```
OI_Research/
â”œâ”€â”€ run.py                          â† START HERE â€” run any experiment
â”‚
â”œâ”€â”€ simulations/
â”‚   â”œâ”€â”€ neuron_models.py            â† LIF, AdEx, Izhikevich, Hodgkin-Huxley + GPU pop
â”‚   â”œâ”€â”€ stdp.py                     â† Classical, Multiplicative, Triplet, R-STDP + GPU layer
â”‚   â”œâ”€â”€ reservoir.py                â† ESN, LSM (spiking), OrganoidReservoir, Analyzer
â”‚   â””â”€â”€ pong_experiment.py          â† DishBrain Pong replication (ESN + LSM controllers)
â”‚
â”œâ”€â”€ experiments/
â”‚   â”œâ”€â”€ scaling_study.py            â† ORIGINAL RESEARCH: OI scaling laws
â”‚   â””â”€â”€ results/                    â† All plots and data saved here
â”‚
â”œâ”€â”€ theory/
â”‚   â”œâ”€â”€ open_problems.md            â† 6 open problems with our research angles
â”‚   â”œâ”€â”€ free_energy_bridge.md       â† FEP â†” R-STDP theoretical bridge (draft)
â”‚   â””â”€â”€ hybrid_architecture.md     â† Biological-silicon hybrid architecture spec
â”‚
â””â”€â”€ data/                           â† For real organoid recordings (FinalSpark, Allen)
```

---

## Results

### Organoid Scaling Laws (Original Research)
![Scaling Laws](experiments/results/scaling/scaling_preprint.png)
*Memory Capacity scales as MC ~ N^0.477 â€” more favorable than LLM scaling (Chinchilla: N^-0.076)*

### DishBrain Pong Replication
![Pong Learning Curve](experiments/results/pong_esn_v2.png)
*ESN+FORCE controller achieves 67.4% hit rate â€” genuine learning from sparse reward signal*

---

## Quick Start

```powershell
cd C:\Users\tyson\Documents\OI_Research
.\.venv\Scripts\Activate.ps1

# Run individual experiments
python run.py neurons       # See LIF, AdEx, Izhikevich neuron behavior
python run.py stdp          # See STDP learning windows + weight evolution
python run.py reservoir     # Echo State Network + OrganoidReservoir
python run.py pong          # DishBrain Pong (ESN version)
python run.py pong --lsm    # DishBrain Pong (spiking neurons)
python run.py pong --compare # ESN vs LSM comparison
python run.py scaling        # Scaling laws study (5-15 min)
python run.py all           # Full suite
```

---

## What's Implemented

### Neuron Models (`simulations/neuron_models.py`)
| Model | Biological Accuracy | Speed | Use Case |
|---|---|---|---|
| LIF | Low | Very fast | Large networks, baseline |
| AdEx | High | Medium | Organoid burst dynamics |
| Izhikevich | Very high | Fast | Best balance |
| Hodgkin-Huxley | Gold standard | Slow | Validation |
| GPU Population | LIF-based | Very fast (CUDA) | 10k+ neuron populations |

### Learning Rules (`simulations/stdp.py`)
| Rule | Description |
|---|---|
| Classical STDP | Additive weight changes |
| Multiplicative STDP | Weight-dependent, more stable |
| R-STDP | Reward-gated â€” the DishBrain mechanism |
| GPU STDP Layer | PyTorch layer with online STDP updates |

### Reservoir Computing (`simulations/reservoir.py`)
| System | Description |
|---|---|
| Echo State Network | Classical ESN, benchmark baseline |
| Liquid State Machine | Spiking neurons, biologically accurate |
| OrganoidReservoir | AdEx + R-STDP + MEA interface model |
| ReservoirAnalyzer | ESP, Memory Capacity, Kernel Quality metrics |

### Experiments
| Experiment | Status | Description |
|---|---|---|
| DishBrain Pong | âœ… Complete | Replication of Kagan 2022 |
| Scaling Laws | âœ… Complete | Original research, power law fits |
| FEP Bridge | ðŸ”¬ In progress | Theoretical, mathematical |
| Hybrid Architecture | ðŸ“ Design phase | Spec written, simulation pending |

---

## Open Problems We're Working On

See `theory/open_problems.md` for full details. The three highest priority:

1. **Encoding Problem** â€” Optimal input â†’ MEA stimulation mapping
2. **FEP-STDP Bridge** â€” Proving R-STDP = active inference
3. **Scaling Laws** â€” Do organoids scale like LLMs? (We're measuring this)

---

## Tech Stack

- **Python 3.9** + virtual environment
- **PyTorch 2.8 + CUDA 12.8** â€” GPU-accelerated SNN simulation on RTX 5070 Ti
- **snntorch 0.9** â€” Spiking neural network library
- **Brian2** â€” Biophysical neural simulation
- **NumPy/SciPy** â€” Numerical computation
- **Matplotlib** â€” Visualization
- **Elephant/NEO** â€” Neural data analysis (MEA format compatibility)

---

## Roadmap

```
NOW (May 2026)
  âœ… Neuron models (LIF, AdEx, Izh, HH)
  âœ… STDP rules (Classical, Mult, R-STDP, GPU)
  âœ… Reservoir systems (ESN, LSM, Organoid)
  âœ… DishBrain Pong replication
  âœ… Scaling laws experiment
  âœ… Theory documents (3 open problems developed)

MONTH 2
  â–¡ Run full scaling study, analyze results
  â–¡ Work through FEP-STDP math proof
  â–¡ Implement Î¦ (integrated information) calculator
  â–¡ Apply for FinalSpark research access

MONTH 3
  â–¡ Draft arXiv preprint on scaling laws
  â–¡ First real organoid experiment (FinalSpark)
  â–¡ Begin hybrid architecture simulation
```

---

## Key Papers to Read

1. Kagan et al. 2022 â€” "In vitro neurons learn and exhibit sentience..." (DishBrain)
2. Smirnova et al. 2023 â€” OI Roadmap (Johns Hopkins)
3. Friston 2010 â€” Free Energy Principle
4. Maass 2002 â€” Liquid State Machines
5. Pfister & Gerstner 2006 â€” Triplet STDP

---

*Built with Claude Code | Research companion: Claude Sonnet 4.6*
