"""
Iterasyon 80: N_crit ile CAI ARASINDA BAGLILIK VAR MI?

Donusum: iter 77-79 VC-dimension'u yeniden kesifetti. Yeni degil.

GERCEKTEN NOVEL SORU:
  CAI = "fonksiyon ogrenmek kac adim suruyor" (training steps to convergence)
  N_crit = "fonksiyon TEKLIGI icin minimum ornek sayisi"

Bu ikisi FARKLI seyleri olcuyor:
  CAI: HIZINI olcuyor (kac step)
  N_crit: BELIRSIZLIGINI olcuyor (kac ornek lazim ki modeller anlassin)

Hipotez: CAI ve N_crit KORELELI ama FARKLI boyutlar
  Bir fonksiyon hem yuksek CAI (ogrenilmesi zor) hem dusuk N_crit olabilir
  Veya: dusuk CAI (kolay ogrenilir) ama yuksek N_crit (belirsiz)

Bu hipotez dogruysa: FONKSIYON UZAYININ 2 BOYUTU var:
  Boyut 1: CAI (ogrenme hizi - modelin ozelligi)
  Boyut 2: N_crit (teklik esigi - verinin ozelligi)

Eger sadece 1 boyut (CAI ~ N_crit): birisi gereksiz
Eger 2 bagimsiz boyut: YENI bir kesfin baslangici

DENEY:
  12 farkli fonksiyon icin hem CAI hem N_crit olc
  Spearman korelasyonu hesapla
  Scatter plot analizi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CAIModel(nn.Module):
    def __init__(self, d_in=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)


def measure_cai(fn, X, max_steps=3000, threshold=0.01, n_seeds=3):
    """CAI = median steps to convergence across seeds."""
    Y = fn(X)
    if Y.dim() == 1: Y = Y.unsqueeze(1)
    base_loss = F.mse_loss(torch.zeros_like(Y), Y).item()
    if base_loss < 1e-10: return 0

    steps_list = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = CAIModel(d_in=X.shape[1], seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        converged_at = max_steps
        for step in range(max_steps):
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            if loss.item() < threshold * base_loss:
                converged_at = step
                break
            opt.zero_grad(); loss.backward(); opt.step()
        steps_list.append(converged_at)
    return int(np.median(steps_list))


class NCritModel(nn.Module):
    def __init__(self, d_in=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)


def measure_mt_at_N(fn, N, d_in=10, n_models=5, steps=4000):
    torch.manual_seed(42)
    X_train = torch.randn(N, d_in, device=DEVICE)
    Y_train = fn(X_train).float()
    if Y_train.dim() == 1: Y_train = Y_train.unsqueeze(1)
    torch.manual_seed(999)
    X_test = torch.randn(2000, d_in, device=DEVICE)
    Y_test = fn(X_test).float().squeeze().cpu().numpy()

    mt_corrs = []
    for seed in range(n_models):
        model = NCritModel(d_in=d_in, seed=seed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            loss = nn.MSELoss()(model(X_train), Y_train)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            p = model(X_test).squeeze().cpu().numpy()
        if np.std(p) > 1e-6:
            c = np.corrcoef(p, Y_test)[0,1]
            if not np.isnan(c): mt_corrs.append(abs(c))
    return np.mean(mt_corrs) if mt_corrs else 0.0


# 12 functions: diverse mix of CAI and continuity
functions = {
    'linear':      lambda x: x[:, 0],
    'quadratic':   lambda x: x[:, 0] ** 2,
    'sin':         lambda x: torch.sin(x[:, 0] * 2),
    'sin*x':       lambda x: torch.sin(x[:, 0]) * x[:, 1],
    'product':     lambda x: x[:, 0] * x[:, 1],
    'exp(-x^2)':   lambda x: torch.exp(-x[:, 0]**2),
    'abs':         lambda x: torch.abs(x[:, 0]),
    'step':        lambda x: (x[:, 0] > 0).float() * 2 - 1,
    'xor':         lambda x: ((x[:, 0] > 0) != (x[:, 1] > 0)).float() * 2 - 1,
    '2step':       lambda x: ((x[:, 0] > -1).float() * (x[:, 0] < 1).float()) * 2 - 1,
    '3parity':     lambda x: ((torch.stack([(x[:,i]>0).float() for i in range(3)],dim=1).sum(dim=1)%2)*2-1),
    'relu_combo':  lambda x: torch.relu(x[:, 0]) - torch.relu(x[:, 0] - 1) + 0.5 * torch.relu(x[:, 1]),
}

print("=== Iter 80: N_crit vs CAI -- Are they the same thing? ===")
print(f"Device: {DEVICE}")
print()

# CAI data
print("Measuring CAI for each function...")
torch.manual_seed(42)
X_cai = torch.randn(1000, 10, device=DEVICE)
cai_vals = {}
for name, fn in functions.items():
    cai = measure_cai(fn, X_cai)
    cai_vals[name] = cai
    print(f"  {name:>15s}: CAI={cai}", flush=True)

print()
print("Measuring N_crit proxy (mt_corr at N=300) for each function...")
print("(higher mt_corr at N=300 = lower N_crit = easier)")
mt300_vals = {}
for name, fn in functions.items():
    mt = measure_mt_at_N(fn, N=300)
    mt300_vals[name] = mt
    print(f"  {name:>15s}: mt@N=300={mt:.4f}", flush=True)

# Correlation analysis
print()
print("=== CORRELATION ANALYSIS ===")
names = list(functions.keys())
cai_arr = np.array([cai_vals[n] for n in names], dtype=float)
mt_arr = np.array([mt300_vals[n] for n in names])

# Spearman correlation: CAI vs mt@N=300 (higher mt = lower N_crit)
from scipy.stats import spearmanr
rho, p = spearmanr(cai_arr, mt_arr)
print(f"Spearman(CAI, mt@300): rho={rho:.4f}, p={p:.4f}")
print()

print(f"{'Function':>15s}  {'CAI':>6s}  {'mt@300':>8s}")
print("-" * 34)
for name in sorted(names, key=lambda n: cai_vals[n]):
    print(f"{name:>15s}  {cai_vals[name]:>6d}  {mt300_vals[name]:>8.4f}")

print()
if abs(rho) > 0.7:
    print(f"HIGHLY CORRELATED (rho={rho:.3f}): CAI ~ N_crit, same dimension")
    print("N_crit may just be a re-parameterization of CAI")
elif abs(rho) > 0.4:
    print(f"MODERATE CORRELATION (rho={rho:.3f}): related but different")
    print("CAI and N_crit capture different aspects of difficulty")
else:
    print(f"WEAK CORRELATION (rho={rho:.3f}): DIFFERENT DIMENSIONS!")
    print("CAI = learning speed, N_crit = uniqueness threshold")
    print("Function space has at least 2 independent complexity axes!")

# Identify outliers: high CAI but low N_crit (or vice versa)
print()
print("=== OUTLIER ANALYSIS ===")
print("Functions where CAI and N_crit DISAGREE most:")
cai_norm = (cai_arr - cai_arr.mean()) / (cai_arr.std() + 1e-10)
mt_norm = (mt_arr - mt_arr.mean()) / (mt_arr.std() + 1e-10)
discordance = cai_norm + mt_norm  # high CAI + high mt = discordant
for i in np.argsort(np.abs(discordance))[::-1][:4]:
    n = names[i]
    print(f"  {n:>15s}: CAI={cai_vals[n]:4d}, mt@300={mt300_vals[n]:.3f} (discord={discordance[i]:+.2f})")
