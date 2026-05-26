import json
with open('experiments/results/scaling/scaling_full.json') as f:
    d = json.load(f)

print('=== BASELINE ESN (sr=0.9) ===')
mc = d['esn_baseline']['mc_mean']
kq = d['esn_baseline']['kq_mean']
mc_std = d['esn_baseline']['mc_std']
N  = d['n_range']
for i,n in enumerate(N):
    print(f'  N={n:5d}  MC={mc[i]:.2f}+/-{mc_std[i]:.2f}  KQ={kq[i]:.0f}')

mc_law = d['mc_power_law']
kq_law = d['kq_power_law']
print()
print(f"MC power law: {mc_law['a']:.4f} * N^{mc_law['alpha']:.4f}")
print(f"KQ power law: {kq_law['a']:.4f} * N^{kq_law['alpha']:.4f}")

print()
print('=== EXTRAPOLATION ===')
for name, vals in d['extrapolation'].items():
    print(f"  {name}: MC={vals['mc']:.0f}  KQ={vals['kq']:.0f}")

print()
print('=== SR SWEEP ===')
for sr, res in d['sr_sweep'].items():
    last_mc = res['mc_mean'][-1]
    print(f"  SR={sr}: MC at N={d['n_range'][len(res['mc_mean'])-1]} = {last_mc:.2f}")

print()
print('=== LSM ===')
if 'lsm' in d:
    lsm = d['lsm']
    for i,n in enumerate(lsm['n_range']):
        print(f"  N={n:5d}  MC={lsm['mc_mean'][i]:.1f}  Sep={lsm['sep_mean'][i]:.4f}")
