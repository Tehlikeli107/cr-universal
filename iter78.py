"""
Iterasyon 78: N_crit = a(model) * b^k -- b UNIVERSALMI?

Iter 77: N_crit model boyutuna bagli (Small: 445, Med: 323, Large: 240)
Ama: a(model) degisiyor, peki b (ustel taban) sabit mi?

N_crit(model, k) = a(model) * b(model)^k
Hipotez: b UNIVERSALDIR, a modele ozgu
Test: her model icin k=1,2,3 olc, b'yi ayir

Eger b sabit: k-parity'nin "ustel guclugu" model-agnostik bir sabit
Eger b degisiyor: hem a hem b model-ozgu

Bu bilgi-teorik anlami olan bir soru:
  b = "her boyut dimensionunun ne kadar bilgi gerektirdigini" olcuyor
  Eger b sabit: bu FONKSIYONun ozelligi
  Eger b degisiyor: modelin kapasitesi bunu da etkiliyor
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, d_in=10, hidden=128, n_layers=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def make_parity(k):
    def fn(x, k=k):
        if k == 0: return x[:, 0]
        bits = torch.stack([(x[:, i] > 0).float() for i in range(k)], dim=1)
        return (bits.sum(dim=1) % 2) * 2 - 1
    return fn


def measure_mt_corr(fn, N, model_class, n_models=5, steps=4000):
    torch.manual_seed(42)
    X_train = torch.randn(N, 10, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)
    torch.manual_seed(999)
    X_test = torch.randn(2000, 10, device=DEVICE)
    Y_test = fn(X_test).float()
    y_test_np = Y_test.squeeze().cpu().numpy()

    mt_corrs = []
    for seed in range(n_models):
        model = model_class(seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            loss = nn.MSELoss()(model(X_train), Y_train)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            p = model(X_test).squeeze().cpu().numpy()
        if np.std(p) > 1e-6:
            c = np.corrcoef(p, y_test_np)[0,1]
            if not np.isnan(c): mt_corrs.append(abs(c))
    return np.mean(mt_corrs) if mt_corrs else 0.0


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


def estimate_ncrit(fn, model_class, N_values, n_models=5, steps=4000):
    """Estimate N_crit from sigmoid fit."""
    mts = []
    for N in N_values:
        mt = measure_mt_corr(fn, N, model_class, n_models=n_models, steps=steps)
        mts.append(mt)
    try:
        popt, _ = curve_fit(sigmoid, N_values, mts, p0=[0.003, np.median(N_values)], maxfev=5000)
        return popt[1], np.array(N_values), np.array(mts)
    except Exception:
        return None, np.array(N_values), np.array(mts)


SmallMLP = lambda seed=0: MLP(d_in=10, hidden=32, n_layers=2, seed=seed)
MedMLP = lambda seed=0: MLP(d_in=10, hidden=128, n_layers=3, seed=seed)
LargeMLP = lambda seed=0: MLP(d_in=10, hidden=512, n_layers=4, seed=seed)

model_configs = [
    ("Small(32)", SmallMLP, [200, 400, 600, 1000, 1500, 2000]),
    ("Med(128)",  MedMLP,   [150, 300, 500, 800, 1200, 1800]),
    ("Large(512)", LargeMLP, [100, 200, 300, 500, 800, 1200]),
]

k_values = [1, 2, 3]

print("=== Iter 78: Is the base b in N_crit ~ a*b^k universal? ===")
print(f"Device: {DEVICE}")
print()

ncrit_table = {}  # {model_name: {k: ncrit}}

for model_name, model_cls, N_vals in model_configs:
    print(f"\n--- {model_name} ---")
    ncrit_table[model_name] = {}
    for k in k_values:
        fn = make_parity(k)
        ncrit, Ns, mts = estimate_ncrit(fn, model_cls, N_vals)
        print(f"  k={k}: ", end="", flush=True)
        for N, mt in zip(Ns, mts):
            print(f"N={N}:{mt:.3f} ", end="", flush=True)
        if ncrit:
            print(f"=> N_crit~{ncrit:.0f}")
        else:
            print(f"=> fit failed")
        ncrit_table[model_name][k] = ncrit

print("\n\n=== EXPONENTIAL FIT: b per model ===")
print(f"{'Model':>12s}  {'b (base)':>10s}  {'a (coeff)':>10s}  {'R^2':>6s}")
print("-" * 44)

b_values = []
for model_name in [c[0] for c in model_configs]:
    kv = k_values
    nc = [ncrit_table[model_name].get(k) for k in kv]
    if None not in nc:
        x = np.array(kv, dtype=float)
        y = np.array(nc, dtype=float)
        log_y = np.log(y)
        coeffs = np.polyfit(x, log_y, 1)
        b = np.exp(coeffs[0])
        a = np.exp(coeffs[1])
        y_pred = a * b**x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        print(f"{model_name:>12s}  {b:>10.3f}  {a:>10.1f}  {r2:>6.4f}")
        b_values.append(b)
    else:
        print(f"{model_name:>12s}  N/A (missing data)")

print()
if len(b_values) >= 2:
    b_spread = max(b_values) - min(b_values)
    b_mean = np.mean(b_values)
    b_cv = b_spread / b_mean
    print(f"b values: {[f'{b:.3f}' for b in b_values]}")
    print(f"Spread: {b_spread:.3f} ({b_cv*100:.1f}% of mean)")
    if b_cv < 0.15:
        print("VERDICT: b IS UNIVERSAL (< 15% variation)")
        print(f"  b ~ {b_mean:.2f} is a property of k-parity, not the learner!")
        print(f"  Each boundary dimension multiplies sample complexity by {b_mean:.1f}x")
        print(f"  -> N_crit(k) = a(model) * {b_mean:.1f}^k (UNIVERSAL BASE)")
    else:
        print(f"VERDICT: b varies with model (cv={b_cv:.2f})")
        print("  Both coefficient and exponent are learner-dependent")
