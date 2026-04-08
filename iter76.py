"""
Iterasyon 76: FONKSIYON TEKLIGINDE FAZ GECISI VAR MI?

Iter 75: N_crit ~ 72 * 2.87^k

YENI SORU: mt_corr(N) egrisinin SEKLI ne?
  - Yavash artan (lineer): faz gecisi yok
  - S-sigmoidali (keskin gecis): FAZ GECISI var
  - Ani sivrilme (threshold'da atlama): 1. derece faz gecisi

Faz gecisi olursa:
  - N < N_crit: mt_corr kucuk, sonra ANIDEN ATLIYOR
  - N > N_crit: mt_corr yuksek ve stabil

DENEY: k=2 (xor) icin N=50..5000 arasi 20 nokta al, egriyi ciz.
Egri: S-sigmoidal mi, lineer mi, yoksa gecikme+ani atlama mi?

Eger FAZ GECISI: yeni fizik analogu -> "bilgi kristalessimi"
"""
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, d_in=10, hidden=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)


def evaluate_k(fn, N, d_in=10, n_models=6, steps=5000):
    torch.manual_seed(42)
    X_train = torch.randn(N, d_in, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)

    torch.manual_seed(999)
    X_test = torch.randn(2000, d_in, device=DEVICE)
    Y_test = fn(X_test).float()
    y_test_np = Y_test.squeeze().cpu().numpy()

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
            c = np.corrcoef(p, y_test_np)[0,1]
            if not np.isnan(c): mt_corrs.append(abs(c))

    return np.mean(mt_corrs) if mt_corrs else 0.0


# k=2 (XOR) - test many N values to find shape of learning curve
def xor_fn(x):
    bits = torch.stack([(x[:, i] > 0).float() for i in range(2)], dim=1)
    return (bits.sum(dim=1) % 2) * 2 - 1

# k=3 (3-parity) - also test to compare
def parity3_fn(x):
    bits = torch.stack([(x[:, i] > 0).float() for i in range(3)], dim=1)
    return (bits.sum(dim=1) % 2) * 2 - 1

# Sample sizes: log-spaced from 50 to 10000
N_values_k2 = [50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000]
N_values_k3 = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]

print("=== Iter 76: Phase Transition in Function Uniqueness? ===")
print(f"Device: {DEVICE}")
print("\nTesting k=2 (XOR) learning curve shape...")
print(f"{'N':>6s}  {'mt_corr':>8s}  {'delta':>8s}")
print("-" * 28)

results_k2 = []
prev = 0
for N in N_values_k2:
    mt = evaluate_k(xor_fn, N)
    delta = mt - prev
    results_k2.append((N, mt))
    marker = " <-- JUMP" if delta > 0.1 else ""
    print(f"{N:>6d}  {mt:>8.4f}  {delta:>+8.4f}{marker}", flush=True)
    prev = mt

print("\n\nTesting k=3 (3-parity) learning curve shape...")
print(f"{'N':>6s}  {'mt_corr':>8s}  {'delta':>8s}")
print("-" * 28)

results_k3 = []
prev = 0
for N in N_values_k3:
    mt = evaluate_k(parity3_fn, N)
    delta = mt - prev
    results_k3.append((N, mt))
    marker = " <-- JUMP" if delta > 0.1 else ""
    print(f"{N:>6d}  {mt:>8.4f}  {delta:>+8.4f}{marker}", flush=True)
    prev = mt

# Analyze curve shapes
print("\n\n=== CURVE SHAPE ANALYSIS ===")

def analyze_curve(data, label):
    Ns = np.array([d[0] for d in data])
    mts = np.array([d[1] for d in data])

    # Find maximum jump
    deltas = np.diff(mts)
    max_jump_idx = np.argmax(deltas)
    max_jump = deltas[max_jump_idx]
    jump_N = Ns[max_jump_idx]

    # Fit sigmoid: y = 1 / (1 + exp(-a*(x-b)))
    from scipy.optimize import curve_fit
    try:
        def sigmoid(x, a, b):
            return 1 / (1 + np.exp(-a * (x - b)))
        popt, _ = curve_fit(sigmoid, Ns, mts, p0=[0.001, 1000], maxfev=5000)
        a_fit, b_fit = popt
        y_pred = sigmoid(Ns, *popt)
        ss_res = np.sum((mts - y_pred)**2)
        ss_tot = np.sum((mts - np.mean(mts))**2)
        r2_sigmoid = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    except Exception:
        r2_sigmoid = 0
        a_fit, b_fit = 0, 0

    # Fit linear
    lin_coeffs = np.polyfit(Ns, mts, 1)
    lin_pred = np.polyval(lin_coeffs, Ns)
    ss_res_lin = np.sum((mts - lin_pred)**2)
    r2_linear = 1 - ss_res_lin/ss_tot if ss_tot > 0 else 0

    print(f"\n{label}:")
    print(f"  Max jump: delta={max_jump:.4f} at N={jump_N}")
    print(f"  Sigmoid fit: R^2={r2_sigmoid:.4f}, midpoint N={b_fit:.0f}")
    print(f"  Linear fit: R^2={r2_linear:.4f}")
    if r2_sigmoid > r2_linear + 0.05:
        print(f"  VERDICT: SIGMOID shape -> PHASE TRANSITION detected!")
        print(f"  Critical N ~ {b_fit:.0f}")
    elif max_jump > 0.15:
        print(f"  VERDICT: ABRUPT JUMP at N={jump_N} -> 1st order phase transition?")
    else:
        print(f"  VERDICT: Gradual/linear increase, no sharp transition")

analyze_curve(results_k2, "k=2 (XOR)")
analyze_curve(results_k3, "k=3 (3-parity)")

print("\n\n=== CONCLUSION ===")
print("If sigmoid: 'crystallization' of function -- N_crit is critical temperature")
print("If linear: smooth emergence, no sharp phase transition")
print("If abrupt jump: 1st order transition (like magnetization)")
print()
print("Physical analogy:")
print("  N < N_crit: 'liquid' (many possible functions)")
print("  N = N_crit: 'crystallization point'")
print("  N > N_crit: 'crystal' (one specific function)")
