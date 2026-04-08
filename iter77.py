"""
Iterasyon 77: N_crit FONKSIYONUN MU, OGRENICININ MI?

Iter 76: k=2 XOR icin N_crit ~ 352, k=3 icin N_crit ~ 1397

SORU: Bu N_crit sabit mi?
  Hipotez A: N_crit FONKSIYONA OZGU (XOR'un inherent ozelliği)
  Hipotez B: N_crit MODELE OZGU (MLP mimarisine bagli)
  Hipotez C: N_crit VERIYE OZGU (dagilima bagli)

Test A: Kucuk MLP (32 hidden) vs Buyuk MLP (512 hidden)
  Eger N_crit degismiyorsa -> FONKSIYONA OZGU (model-agnostic)
  Eger N_crit kucuk MLP icin yukseliyor -> MODELE OZGU

Test B: Gaussian vs Uniform veri dagilimlari
  Eger N_crit degismiyorsa -> FONKSIYONA OZGU (distribution-agnostic)

Test C: Tamamen farkli mimari: CNN vs MLP
  k=2 XOR odakli, 1D konvolüsyon yerine attention mekanizmasi

TAHMIN:
  N_crit MODELDEN BAGIMSIZ ama FONKSIYONA OZGU.
  Cunku N_crit ozu, fonksiyonun kac ornekle BELIRLENEBILDIGINI olcuyor.
  Bu fonksiyonun bilgi-teorik ozelligi, ogrenicinin degil.
"""
import torch
import torch.nn as nn
import numpy as np

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


def measure_mt_corr(fn, N, model_class, d_in=10, n_models=6, steps=5000):
    """Model-target correlation on held-out test set."""
    torch.manual_seed(42)
    X_train = torch.randn(N, d_in, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)

    torch.manual_seed(999)
    X_test = torch.randn(2000, d_in, device=DEVICE)
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


def xor_fn(x):
    bits = torch.stack([(x[:, i] > 0).float() for i in range(2)], dim=1)
    return (bits.sum(dim=1) % 2) * 2 - 1


# Model variants
SmallMLP = lambda seed=0: MLP(d_in=10, hidden=32, n_layers=2, seed=seed)
MedMLP = lambda seed=0: MLP(d_in=10, hidden=128, n_layers=3, seed=seed)
LargeMLP = lambda seed=0: MLP(d_in=10, hidden=512, n_layers=4, seed=seed)

# Key N values around N_crit ~ 352
N_values = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]

print("=== Iter 77: Is N_crit Function-Specific or Learner-Specific? ===")
print(f"Device: {DEVICE}")
print("\nTest: XOR function, 3 different MLP sizes")
print("Question: Does N_crit change with model capacity?")
print()

results = {
    'Small(32)': [],
    'Med(128)': [],
    'Large(512)': [],
}

print(f"{'N':>6s}  {'Small(32)':>10s}  {'Med(128)':>10s}  {'Large(512)':>11s}")
print("-" * 48)

for N in N_values:
    s = measure_mt_corr(xor_fn, N, SmallMLP)
    m = measure_mt_corr(xor_fn, N, MedMLP)
    l = measure_mt_corr(xor_fn, N, LargeMLP)
    results['Small(32)'].append((N, s))
    results['Med(128)'].append((N, m))
    results['Large(512)'].append((N, l))
    print(f"{N:>6d}  {s:>10.4f}  {m:>10.4f}  {l:>11.4f}", flush=True)


# Fit sigmoid to each and extract N_crit
from scipy.optimize import curve_fit

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

print("\n\n=== SIGMOID FIT: N_crit for each model ===")
ncrit_by_model = {}
for name, data in results.items():
    Ns = np.array([d[0] for d in data], dtype=float)
    mts = np.array([d[1] for d in data])
    try:
        popt, _ = curve_fit(sigmoid, Ns, mts, p0=[0.003, 500], maxfev=5000)
        a_fit, b_fit = popt
        print(f"  {name}: N_crit ~ {b_fit:.0f}")
        ncrit_by_model[name] = b_fit
    except Exception as e:
        print(f"  {name}: fit failed ({e})")
        ncrit_by_model[name] = None

print()
crits = [v for v in ncrit_by_model.values() if v is not None]
if len(crits) >= 2:
    spread = max(crits) - min(crits)
    mean_crit = np.mean(crits)
    cv = spread / mean_crit  # coefficient of variation
    print(f"  N_crit values: {[f'{c:.0f}' for c in crits]}")
    print(f"  Spread: {spread:.0f} ({cv*100:.1f}% of mean)")
    if cv < 0.2:
        print("  VERDICT: N_crit is MODEL-INDEPENDENT (< 20% variation)")
        print("  -> N_crit is a property of the FUNCTION, not the learner!")
    else:
        print(f"  VERDICT: N_crit VARIES with model (cv={cv:.2f})")
        print("  -> N_crit depends on both function AND learner")

print()
print("=== THEORETICAL IMPLICATION ===")
print("If model-independent:")
print("  N_crit = inherent 'sample complexity' of the function")
print("  Like Vapnik-Chervonenkis dimension but for uniqueness, not generalization")
print("  N_crit(k-parity) ~ 4^k is a FUNDAMENTAL CONSTANT of the function class")
print()
print("If model-dependent:")
print("  N_crit measures learner capacity, not function complexity")
print("  Different models would learn different 'versions' of the function")
print("  Less interesting -- just characterizes overfitting")
