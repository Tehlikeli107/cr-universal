"""
Iterasyon 79: KRITIK USSEL BETA -- FAZ GECISININ DERECESI

Iter 76: Ogrenme egrisi sigmoidal (2. derece faz gecisi)
Iter 78: N_crit(k) ~ a * 3.8^k

YENI SORU: Sigmoid'in SEKLI evrensel mi?
Faz gecisi teorisi: kritik noktada order parameter:
  mt_corr(N) ~ (N/N_crit - 1)^beta  (N > N_crit icin)

Eger beta sabit (k ve model bagimsiz): UNIVERSALLIK var
  - Farkli "universalite siniflari" farkli beta verir
  - Ising modeli: beta=1/8 (2D), beta=1/2 (mean field)
  - Bizim sistem: hangi sinifta?

DENEY:
  k=2 (XOR) ve k=3 (parity) icin N_crit etrafinda yogun olcum
  log(mt_corr) ~ beta * log(N/N_crit - 1) dogrusal mi?
  Beta degeri nedir?

TAHMIN:
  Eger smooth S-sigmoid: beta ~ 1 (mean field)
  Eger keskin: beta < 1 (daha hizli gecis)
  Eger yavash: beta > 1
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


def measure_mt(fn, N, n_models=6, steps=5000, d_in=10):
    torch.manual_seed(42)
    X_train = torch.randn(N, d_in, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)
    torch.manual_seed(999)
    X_test = torch.randn(2000, d_in, device=DEVICE)
    Y_test = fn(X_test).float().squeeze().cpu().numpy()

    preds = []
    for seed in range(n_models):
        model = MLP(d_in=d_in, seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            loss = nn.MSELoss()(model(X_train), Y_train)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            p = model(X_test).squeeze().cpu().numpy()
        preds.append(p)

    mt_corrs = []
    for p in preds:
        if np.std(p) > 1e-6:
            c = np.corrcoef(p, Y_test)[0,1]
            if not np.isnan(c): mt_corrs.append(abs(c))
    return np.mean(mt_corrs) if mt_corrs else 0.0


def xor_fn(x):
    bits = torch.stack([(x[:, i] > 0).float() for i in range(2)], dim=1)
    return (bits.sum(dim=1) % 2) * 2 - 1

def parity3_fn(x):
    bits = torch.stack([(x[:, i] > 0).float() for i in range(3)], dim=1)
    return (bits.sum(dim=1) % 2) * 2 - 1

# From iter 76: N_crit(k=2) ~ 352, N_crit(k=3) ~ 1397
# Measure densely around N_crit
N_vals_k2 = [150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500]
N_vals_k3 = [500, 700, 1000, 1200, 1400, 1700, 2000, 2500, 3000, 4000]

print("=== Iter 79: Critical Exponent Beta ===")
print(f"Device: {DEVICE}")
print()

def fit_critical_exponent(N_vals, N_crit_estimate, label):
    """Fit: mt_corr ~ (N/N_crit - 1)^beta near N_crit"""
    print(f"Measuring {label} (N_crit ~ {N_crit_estimate}):")
    Ns = []
    mts = []
    for N in N_vals:
        mt = measure_mt(xor_fn if "k=2" in label else parity3_fn, N)
        print(f"  N={N:5d}: mt={mt:.4f}", flush=True)
        Ns.append(N)
        mts.append(mt)

    Ns = np.array(Ns, dtype=float)
    mts = np.array(mts)

    # Use only points with N > N_crit (above transition)
    above = Ns > N_crit_estimate
    if above.sum() < 3:
        print("  Not enough points above N_crit for fit")
        return None, None

    x = (Ns[above] / N_crit_estimate - 1)
    y = mts[above]

    # Fit power law: y ~ A * x^beta
    # Log-log: log(y) = log(A) + beta * log(x)
    valid = (x > 0) & (y > 0.05)  # need y > 0 for log
    if valid.sum() < 3:
        print("  Not enough valid points for log-log fit")
        return None, None

    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    coeffs = np.polyfit(log_x, log_y, 1)
    beta = coeffs[0]
    A = np.exp(coeffs[1])

    rho = np.corrcoef(log_x, log_y)[0,1]

    print(f"\n  Power law fit (above N_crit):")
    print(f"  mt_corr ~ {A:.3f} * (N/N_crit - 1)^{beta:.3f}")
    print(f"  log-log R={rho:.4f}")
    print(f"  beta = {beta:.3f}")

    return beta, rho


print("=== k=2 (XOR): critical exponent ===")
beta_k2, rho_k2 = fit_critical_exponent(N_vals_k2, 352, "k=2")
print()

print("=== k=3 (3-parity): critical exponent ===")
beta_k3, rho_k3 = fit_critical_exponent(N_vals_k3, 1397, "k=3")
print()

print("=== UNIVERSALITY CHECK ===")
if beta_k2 and beta_k3:
    print(f"  beta(k=2) = {beta_k2:.3f}")
    print(f"  beta(k=3) = {beta_k3:.3f}")
    diff = abs(beta_k2 - beta_k3) / np.mean([beta_k2, beta_k3])
    print(f"  Relative difference: {diff*100:.1f}%")
    if diff < 0.2:
        print("  UNIVERSAL: beta consistent across k (< 20% variation)")
        print(f"  Universality class: beta ~ {np.mean([beta_k2, beta_k3]):.2f}")
        print()
        print("  Physical interpretation:")
        print(f"  beta=0.5 -> mean field (Ising above d=4)")
        print(f"  beta=0.33 -> 3D Ising")
        print(f"  beta=0.125 -> 2D Ising (Onsager)")
        print(f"  Our beta={np.mean([beta_k2, beta_k3]):.2f}: what class?")
    else:
        print("  NOT universal: beta varies with k")
        print("  Phase transition type may depend on function complexity")
else:
    print("  Could not estimate beta for both k values")

print("\n=== CONCLUSION ===")
print("If beta universal: machine learning has universal critical exponents!")
print("  - Like statistical mechanics, ML exhibits universality classes")
print("  - N_crit is the 'critical temperature' of function learning")
print("  - Below N_crit: 'disordered' (many functions possible)")
print("  - Above N_crit: 'ordered' (unique function emerges)")
