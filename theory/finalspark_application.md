# FinalSpark Research Access Application
*Ready to send — edit [bracketed] fields before sending*

**To:** research@finalspark.com
**Subject:** Research Access Request — Organoid Scaling Laws + FEP-STDP Bridge

---

Dear FinalSpark Research Team,

I am writing to request research access to the Neuroplatform to validate
a computational model I have been developing over the past several months.

**Background**

I am an independent researcher working at the intersection of computational
neuroscience, reservoir computing theory, and organoid intelligence. My work
is motivated by the DishBrain experiment (Kagan et al. 2022) and the OI
roadmap published by Smirnova et al. (2023).

**What I've built**

I have developed a full simulation stack for organoid intelligence research:

1. **Biophysical neuron models** — LIF, AdEx, Izhikevich, and Hodgkin-Huxley
   implementations with GPU-accelerated population simulation (CUDA)

2. **OrganoidReservoir** — A MEA-coupled reservoir model using AdEx neurons,
   reward-modulated STDP (R-STDP), and biologically heterogeneous parameters.
   The architecture directly mirrors the FinalSpark neuroplatform interface.

3. **DishBrain Pong replication** — Computational replication of the Kagan 2022
   experiment using both Echo State Network and Liquid State Machine controllers
   with FORCE learning, currently achieving >65% hit rate in simulation.

4. **Scaling laws study** — An original research experiment measuring memory
   capacity (MC) and kernel quality (KQ) as power functions of reservoir size N.
   Preliminary results: MC ~ N^0.877, which is more favorable scaling than
   large language models (Chinchilla: loss ~ N^-0.076). This has not been
   previously characterized in the literature.

5. **FEP-STDP bridge** — A formal theoretical derivation showing that
   reward-modulated STDP is gradient descent on variational free energy
   (Friston's Free Energy Principle). This provides a principled explanation
   for why the chaos/ordered stimulation protocol in DishBrain works.

**What I want to do with Neuroplatform access**

My simulation makes three concrete, testable predictions about real organoid
behavior:

**P1 (Precision-learning rate):** Learning rate should scale with MEA SNR.
Higher-quality electrode contact → faster behavioral adaptation.
*Test: vary stimulation amplitude, measure learning curve slope.*

**P2 (Spontaneous activity as prior):** Between stimulation trials, population
activity should cluster near recently-learned task patterns.
*Test: record spontaneous activity at rest, compare to task-period patterns.*

**P3 (Optimal stimulation contrast):** Learning speed should increase with
the entropy contrast between "reward" and "punishment" stimulation patterns.
*Test: vary chaos-to-ordered entropy ratio, measure learning rate.*

Each prediction is derived from the FEP-STDP equivalence and can be tested
with standard Neuroplatform API calls — no special wetlab protocols required.

**My setup**

- AMD Ryzen 7 7800X3D + RTX 5070 Ti (16GB GDDR7, CUDA 12.8)
- Full simulation codebase (Python/PyTorch/snntorch/Brian2)
- Working scaling study with preliminary publishable results
- Theoretical framework for FEP-STDP equivalence in progress

I am working toward a preprint on the scaling laws result (target: Q3 2026)
and would greatly value being able to include a section validating the
computational model against real organoid recordings from your platform.

I am happy to share my full codebase, discuss collaboration, or adjust the
experimental design to fit your platform's current capabilities and ethical
guidelines.

Thank you for your time and for the remarkable work your team has done
in making organoid computing accessible to the research community.

Best regards,
Tyson
tysonguerrero96@gmail.com
[Optional: LinkedIn / GitHub if you have one]

---

**Attachments to include when sending:**
- [ ] Screenshot of `experiments/results/scaling/scaling_preprint.png`
- [ ] Screenshot of Pong learning curve
- [ ] Link to GitHub repo if you create one (suggested: `tyson-oi-research`)
- [ ] 1-page PDF summary of the FEP-STDP derivation (export from theory doc)

---

**Follow-up if no response in 2 weeks:**
Re-send to: contact@finalspark.com with subject line
"Follow-up: Independent Research Access — Organoid Scaling Study"
