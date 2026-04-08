"""
Iterasyon 75c: N_CRIT -- DOGRU OLCUM

Problem: modeller ogrenemiyorsa hepsi ~0 tahmin ediyor, pairwise corr=1.0 YANLIS.
Cozum: TEST verisi uzerinde olc + model-target correlation da kontrol et.

Dogru uniqueness olcusu:
  1. Modeli egit (train data)
  2. FARKLI test datasinda tahmin al
  3. Test'te model-target correlation = "learning success"
  4. Test'te inter-model correlation = "agreement"

  N_crit = min N ki (avg model-target corr > 0.9) AND (inter-model corr > 0.99)

Eger modeller ogrenemiyorsa: model-target corr DUSUK -> N_crit'e sayilmaz.
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


def evaluate(fn, N_train, k, d_in=10, n_models=6, steps=5000):
    """
    Returns:
      model_target_corr: avg correlation of each model with true target (on test set)
      inter_model_corr: avg pairwise correlation between models (on test set)
    """
    torch.manual_seed(42)
    X_train = torch.randn(N_train, d_in, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)

    # Test set: larger, different seed
    torch.manual_seed(999)
    X_test = torch.randn(2000, d_in, device=DEVICE)
    Y_test = fn(X_test).float()
    if Y_test.dim() == 1: Y_test = Y_test.unsqueeze(1)
    y_test_np = Y_test.squeeze().cpu().numpy()

    preds_test = []
    train_losses = []
    for seed in range(n_models):
        model = MLP(d_in=d_in, seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            loss = nn.MSELoss()(model(X_train), Y_train)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        train_losses.append(loss.item())
        with torch.no_grad():
            p = model(X_test).squeeze().cpu().numpy()
        preds_test.append(p)

    # Model-target correlation
    mt_corrs = []
    for p in preds_test:
        if np.std(p) > 1e-6 and np.std(y_test_np) > 1e-6:
            c = np.corrcoef(p, y_test_np)[0,1]
            if not np.isnan(c): mt_corrs.append(c)
    avg_mt = np.mean(mt_corrs) if mt_corrs else 0.0

    # Inter-model correlation
    im_corrs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            if np.std(preds_test[i]) > 1e-6 and np.std(preds_test[j]) > 1e-6:
                c = np.corrcoef(preds_test[i], preds_test[j])[0,1]
                if not np.isnan(c): im_corrs.append(c)
    avg_im = np.mean(im_corrs) if im_corrs else 0.0

    return avg_mt, avg_im, np.mean(train_losses)


def make_parity(k):
    if k == 0:
        return lambda x: x[:, 0], "0b_linear"
    elif k == 1:
        return lambda x: (x[:, 0] > 0).float() * 2 - 1, "1b_sign"
    else:
        def fn(x, k=k):
            bits = torch.stack([(x[:, i] > 0).float() for i in range(k)], dim=1)
            return (bits.sum(dim=1) % 2) * 2 - 1  # {-1, +1}
        return fn, f"{k}b_parity"


print("=== Iter 75c: N_CRIT with CORRECT METRIC ===")
print(f"Device: {DEVICE}")
print("Metric: model-target corr AND inter-model corr (on held-out test set)")
print()

# Test plan
test_plan = [
    (0, [10, 20, 50, 100]),
    (1, [10, 20, 50, 100, 200]),
    (2, [50, 100, 200, 500, 1000]),
    (3, [100, 200, 500, 1000, 2000]),
    (4, [500, 1000, 2000, 5000]),
    (5, [1000, 2000, 5000, 10000]),
]

ncrit_data = {}

for k, sizes in test_plan:
    fn, label = make_parity(k)
    print(f"k={k} ({label}):")
    ncrit = None
    for N in sizes:
        mt, im, tr = evaluate(fn, N, k)
        print(f"  N={N:6d}: model-target={mt:.4f}, inter-model={im:.4f}, train_loss={tr:.6f}", flush=True)
        # N_crit = models both learn well AND agree
        if mt > 0.90 and im > 0.99:
            ncrit = N
            break
    if ncrit is None:
        ncrit = (f">{sizes[-1]}", sizes[-1])
    ncrit_data[k] = ncrit
    print()

print("=== N_CRIT SUMMARY ===")
numeric_ncrit = []
for k, v in sorted(ncrit_data.items()):
    if isinstance(v, tuple):
        print(f"  k={k}: N_crit > {v[0]}")
        numeric_ncrit.append((k, v[1]))
    else:
        print(f"  k={k}: N_crit ~ {v}")
        numeric_ncrit.append((k, v))

print("\n=== EXPONENTIAL FIT ===")
x = np.array([n for n, _ in numeric_ncrit], dtype=float)
y = np.array([v for _, v in numeric_ncrit], dtype=float)
if len(x) >= 3:
    log_y = np.log(y)
    coeffs = np.polyfit(x, log_y, 1)
    b = np.exp(coeffs[0])
    a = np.exp(coeffs[1])
    rho = np.corrcoef(x, log_y)[0,1]
    print(f"  N_crit ~ {a:.1f} * {b:.2f}^k")
    print(f"  Log-linear rho={rho:.4f}")
    print(f"  b={b:.2f}: each extra dimension MULTIPLIES N_crit by {b:.1f}x")
    print()
    if rho > 0.97:
        print("  CONFIRMED: N_crit grows EXPONENTIALLY with boundary dimensions")
        print(f"  NEW FORMULA: N_crit(k) ~ {a:.0f} * {b:.1f}^k")
    elif rho > 0.90:
        print("  PARTIAL: subexponential growth, maybe power law?")
    else:
        print("  NOT EXPONENTIAL: try other fit")
