import sys, os
sys.path.insert(0, '.')
os.makedirs('experiments/results', exist_ok=True)
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless run

from experiments.scaling_study import run_scaling_study
results, fits = run_scaling_study(
    n_range=[10, 25, 50, 100, 200, 400],
    n_seeds=2,
    max_delay=20
)
mc_alpha = fits["mc_power_law"]["alpha"]
kq_alpha = fits["kq_power_law"]["alpha"]
print(f'\nScaling study complete.')
print(f'MC power law: N^{mc_alpha:.3f}')
print(f'KQ power law: N^{kq_alpha:.3f}')
