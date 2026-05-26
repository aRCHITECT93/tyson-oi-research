"""
run.py — OI Research Project Launcher
======================================
Central entry point for all experiments.

Usage:
  python run.py --help
  python run.py neurons          # Visualize neuron models
  python run.py stdp             # STDP learning rules
  python run.py reservoir        # Reservoir computing foundation
  python run.py pong             # DishBrain Pong (ESN)
  python run.py pong --lsm       # DishBrain Pong (spiking LSM)
  python run.py pong --compare   # ESN vs LSM comparison
  python run.py scaling          # Scaling laws experiment (5-15 min)
  python run.py all              # Run everything in sequence
"""

import argparse
import os
import sys
import time

os.makedirs("experiments/results", exist_ok=True)

def header(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}\n")


def run_neurons():
    header("NEURON MODELS")
    from simulations.neuron_models import LIFNeuron, AdExNeuron, IzhikevichNeuron, GPUNeuronPopulation, IzhikevichParams
    import numpy as np, torch

    dt = 0.1; T = 500
    t = np.arange(0, T, dt)
    I = np.where(t > 100, 3.5, 0.0)

    lif  = LIFNeuron(dt=dt)
    adex = AdExNeuron(dt=dt)
    izh  = IzhikevichNeuron(IzhikevichParams(a=0.02, b=0.2, c=-50.0, d=2.0), dt=0.5)

    V_lif, sp_lif     = lif.simulate(I)
    V_adex, _, sp_adex = adex.simulate(I * 500)
    t2 = np.arange(0, T, 0.5)
    V_izh, sp_izh = izh.simulate(np.where(t2 > 100, 10.0, 0.0))

    print(f"LIF:   {sp_lif.sum()} spikes")
    print(f"AdEx:  {sp_adex.sum()} spikes (burst-capable)")
    print(f"Izh:   {sp_izh.sum()} spikes (chattering)")

    if torch.cuda.is_available():
        pop = GPUNeuronPopulation(n_neurons=10000)
        x = torch.rand(10000).to(pop.device)
        spk, _ = pop(x)
        print(f"GPU pop (10k neurons): {int(spk.sum())} spikes on {pop.device}")

    import subprocess
    subprocess.run([sys.executable, "simulations/neuron_models.py"], check=False)


def run_stdp():
    header("STDP LEARNING RULES")
    import subprocess
    subprocess.run([sys.executable, "simulations/stdp.py"], check=False)


def run_reservoir():
    header("RESERVOIR COMPUTING")
    import subprocess
    subprocess.run([sys.executable, "simulations/reservoir.py"], check=False)


def run_pong(mode="esn"):
    if mode == "compare":
        header("DISHBRAIN PONG — ESN vs LSM COMPARISON")
        from simulations.pong_experiment import compare_controllers
        compare_controllers(n_steps=3000)
    elif mode == "lsm":
        header("DISHBRAIN PONG — LIQUID STATE MACHINE (SPIKING)")
        from simulations.pong_experiment import run_pong_experiment
        run_pong_experiment(n_steps=4000, controller_type="lsm")
    else:
        header("DISHBRAIN PONG — ECHO STATE NETWORK")
        from simulations.pong_experiment import run_pong_experiment
        run_pong_experiment(n_steps=4000, controller_type="esn")


def run_scaling():
    header("ORGANOID SCALING LAWS (original research)")
    from experiments.scaling_study import run_scaling_study
    run_scaling_study(n_range=[10, 25, 50, 100, 200, 400, 800], n_seeds=3)


def run_all():
    header("RUNNING FULL EXPERIMENT SUITE")
    t0 = time.time()

    print("[1/4] Neuron models...")
    run_neurons()

    print("\n[2/4] STDP learning...")
    run_stdp()

    print("\n[3/4] Reservoir foundation...")
    run_reservoir()

    print("\n[4/4] DishBrain Pong...")
    run_pong("esn")

    print(f"\n{'═'*60}")
    print(f"  ALL EXPERIMENTS COMPLETE in {time.time()-t0:.0f}s")
    print(f"  Results in: experiments/results/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OI Research — Organoid Intelligence Simulation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("experiment",
                        choices=["neurons", "stdp", "reservoir", "pong", "scaling", "all"],
                        help="Experiment to run")
    parser.add_argument("--lsm", action="store_true", help="Use LSM for pong")
    parser.add_argument("--compare", action="store_true", help="Compare ESN vs LSM for pong")

    args = parser.parse_args()

    if args.experiment == "neurons":  run_neurons()
    elif args.experiment == "stdp":   run_stdp()
    elif args.experiment == "reservoir": run_reservoir()
    elif args.experiment == "pong":
        if args.compare: run_pong("compare")
        elif args.lsm:   run_pong("lsm")
        else:            run_pong("esn")
    elif args.experiment == "scaling": run_scaling()
    elif args.experiment == "all":     run_all()
