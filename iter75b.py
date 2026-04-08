"""
Iterasyon 75b: N_CRIT USTEL HIPOTEZI -- DUZELTILMIS

Iter 74'teki gercek zorluk = BOYUTSAL ETKILESIM.
  sin: 1D smooth -> N_crit=10
  step: 1D discontinuous -> N_crit=50
  xor: 2D (sign(x0) XOR sign(x1)) -> N_crit=1000

1D step fonksiyonlari KOLAY cunku sadece 1 noktayi bulmak yeter.
Asil zorluk: karar sinirinin BOYUTLARI KESEN HIPER-DUZLEM SAYISI.

Yeni fonksiyonlar (k-boyutlu parity):
  k=0: f(x) = x0 (linear, 0 boundary)
  k=1: f(x) = sign(x0) (1 boundary)
  k=2: f(x) = sign(x0) XOR sign(x1) (2 boundaries, XOR)
  k=3: f(x) = (x0>0 + x1>0 + x2>0) % 2 (3 boundaries, 3-parity)
  k=4: f(x) = (x0>0 + x1>0 + x2>0 + x3>0) % 2 (4-parity)
  k=5: f(x) = (x0>0 + ... + x4>0) % 2 (5-parity)

HIPOTEZ: N_crit ~ 10 * C^k (C ~ 5-10 per boundary dimension)
TAHMIN: k=4 -> N_crit ~ 6000-10000, k=5 -> N_crit ~ 30000-100000
"""
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, d_in=10, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)


def measure_uniqueness(fn, N, d_in=10, n_models=8, steps=3000):
    X = torch.randn(N, d_in, device=DEVICE)
    Y = fn(X).float()
    if Y.dim() == 1: Y = Y.unsqueeze(1)

    preds = []
    for seed in range(n_models):
        model = MLP(d_in=d_in, seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(steps):
            loss = nn.MSELoss()(model(X), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = model(X).squeeze().cpu().numpy()
        preds.append(p)

    corrs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            c = np.corrcoef(preds[i], preds[j])[0,1]
            if not np.isnan(c): corrs.append(c)
    return np.mean(corrs) if corrs else 0.0


# Parity functions: n_boundary_dims = number of dimensions checked
def make_parity(k):
    """k-dimensional parity: sum(xi>0 for i<k) % 2"""
    if k == 0:
        return lambda x: x[:, 0], "0b_linear"
    elif k == 1:
        return lambda x: (x[:, 0] > 0).float(), "1b_sign"
    else:
        def fn(x, k=k):
            bits = torch.stack([(x[:, i] > 0).float() for i in range(k)], dim=1)
            return (bits.sum(dim=1) % 2)
        return fn, f"{k}b_parity"


print("=== Iter 75b: N_crit ~ C^k (Dimensional Boundary Hypothesis) ===")
print(f"Device: {DEVICE}")
print("\nHypothesis: N_crit ~ 10 * C^k where k = # boundary dimensions")
print("Prediction: k=4 -> ~6000-10000, k=5 -> ~30000+")
print()

# Test each k with adaptive sample sizes
test_plan = [
    (0, [10, 20, 50]),
    (1, [20, 50, 100]),
    (2, [50, 100, 200, 500, 1000]),
    (3, [100, 200, 500, 1000, 2000]),
    (4, [500, 1000, 2000, 5000, 10000]),
    (5, [2000, 5000, 10000]),  # only test a few due to time
]

all_results = {}
ncrit_estimates = {}

for k, sizes in test_plan:
    fn, label = make_parity(k)
    print(f"Testing k={k} ({label}):")
    results = {}
    ncrit = None
    prev_u = None
    for N in sizes:
        u = measure_uniqueness(fn, N)
        results[N] = u
        print(f"  N={N:6d}: uniq={u:.4f}", flush=True)
        if u > 0.99:
            ncrit = N
            break
        prev_u = u
    all_results[label] = results
    ncrit_estimates[k] = ncrit if ncrit else (f">{sizes[-1]}", sizes[-1])
    print()

print("\n=== N_CRIT SUMMARY ===")
ncrit_values = []
for k, est in sorted(ncrit_estimates.items()):
    if isinstance(est, tuple):
        label = f">{est[0]}"
        val = est[1]
    else:
        label = str(est)
        val = est
    print(f"  k={k}: N_crit ~ {label}")
    ncrit_values.append((k, val))

# Fit
print("\n=== EXPONENTIAL FIT ===")
x = np.array([n for n, _ in ncrit_values], dtype=float)
y = np.array([v for _, v in ncrit_values], dtype=float)

log_y = np.log(y)
coeffs = np.polyfit(x, log_y, 1)
b = np.exp(coeffs[0])
a = np.exp(coeffs[1])
print(f"  N_crit ~ {a:.1f} * {b:.2f}^k")
print(f"  Hypothesis: a~10, b~5. Got: a={a:.1f}, b={b:.2f}")

rho = np.corrcoef(x, log_y)[0,1]
print(f"  Log-linear rho={rho:.4f} (>0.99 = exponential confirmed)")

print("\n=== CONCLUSION ===")
print(f"If b ~ 2-10: N_crit ~ a * b^k (EXPONENTIAL)")
print(f"If rho < 0.95: not exponential, find other formula")
print(f"Key question: does each extra boundary dimension MULTIPLY N_crit by ~b?")
