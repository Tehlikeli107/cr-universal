"""
NEXUS v2: Lamarckian iteration on v1 (which beat transformer 89.2% vs 85.4%).

v1 worked because: explicit relation encoding (diff between tokens in window).
v2 changes (ONE AT A TIME tested):
  1. LEARNED relation function (not hand-crafted diff)
  2. 4 layers instead of 2
  3. Wider windows (up to 12)
  4. Walk-based information flow (A^k) for long-range without attention

Test on same task: mixed rules (copy-2 first half, sum-rule second half).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time
DEVICE = torch.device('cuda')

class LearnedRelation(nn.Module):
    """LEARNED relation function. Not diff/prod - let network decide."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d*2, d*2), nn.GELU(), nn.Linear(d*2, d))
    def forward(self, xi, xj):
        return self.net(torch.cat([xi, xj], dim=-1))

class NexusV2Layer(nn.Module):
    """One NEXUS v2 layer: learned relation + local window + gated aggregation."""
    def __init__(self, d, windows):
        super().__init__()
        self.d = d
        self.windows = windows
        self.n_win = len(windows)
        # One learned relation per window size
        self.relations = nn.ModuleList([LearnedRelation(d) for _ in windows])
        # Combine window outputs
        self.combine = nn.Linear(d * self.n_win, d)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape
        window_outputs = []

        for wi, w in enumerate(self.windows):
            # Pad for window
            xp = F.pad(x, (0,0,w,w))
            wsize = 2*w + 1
            idx = torch.arange(N, device=DEVICE).unsqueeze(1) + torch.arange(wsize, device=DEVICE)
            xw = xp[:, idx]  # [B, N, wsize, D]
            xi = x.unsqueeze(2).expand_as(xw)  # [B, N, wsize, D]

            # LEARNED relation between xi and each window position
            rel = self.relations[wi](xi.reshape(-1, D), xw.reshape(-1, D))
            rel = rel.reshape(B, N, wsize, D)

            # Causal mask
            offsets = torch.arange(wsize, device=DEVICE) - w
            causal = (offsets > 0).unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            rel = rel.masked_fill(causal.unsqueeze(-1), 0)

            # Aggregate: mean of valid positions
            n_valid = (~causal).float().sum(dim=-1, keepdim=True).clamp(min=1)
            agg = rel.sum(dim=2) / n_valid  # [B, N, D]
            window_outputs.append(agg)

        # Combine all windows
        combined = self.combine(torch.cat(window_outputs, dim=-1))
        x = self.norm(x + combined)
        x = self.norm2(x + self.ffn(x))
        return x

class NexusV2(nn.Module):
    """NEXUS v2: 4 layers, increasing windows, learned relations."""
    def __init__(self, vocab, d=64, max_seq=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.layers = nn.ModuleList([
            NexusV2Layer(d, [1, 2, 3]),       # local
            NexusV2Layer(d, [2, 4, 6]),       # medium
            NexusV2Layer(d, [3, 6, 9]),       # wide
            NexusV2Layer(d, [4, 8, 12]),      # very wide
        ])
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        B, N = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(N, device=ids.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

class Transformer(nn.Module):
    def __init__(self, vocab, d=64, nh=4, nl=4, max_seq=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        el = nn.TransformerEncoderLayer(d, nh, d*4, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, nl)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        B,N = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(N, device=ids.device))
        mask = torch.triu(torch.ones(N,N,device=ids.device), 1).bool()
        return self.head(self.enc(x, mask=mask))

def gen_mixed(V=20, seq=32, batch=64):
    t = torch.randint(0, V, (batch, seq), device=DEVICE)
    mid = seq // 2
    for b in range(batch):
        # First half: copy 2 back
        for i in range(2, mid): t[b, i] = t[b, i-2]
        # Second half: sum rule
        for i in range(max(mid, 3), seq): t[b, i] = (t[b, i-1] + t[b, i-3]) % V
    return t[:, :-1], t[:, 1:]

V=20; SEQ=24; EP=100; BS=32
print("NEXUS v2 vs TRANSFORMER (4 layers each)")
print("="*50)
print(f"Mixed rules: copy-2 (first half) + sum-rule (second half)")
print(f"V={V}, seq={SEQ}, epochs={EP}\n")

for name, make in [
    ("Transformer-4L", lambda: Transformer(V, 64, 4, 4, SEQ)),
    ("NEXUS-v2-4L", lambda: NexusV2(V, 64, SEQ)),
]:
    torch.manual_seed(42)
    model = make().to(DEVICE)
    np_ = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(EP):
        inp, tgt = gen_mixed(V, SEQ, BS)
        loss = F.cross_entropy(model(inp).reshape(-1,V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    with torch.no_grad():
        inp, tgt = gen_mixed(V, SEQ, 2000)
        logits = model(inp)
        ppl = F.cross_entropy(logits.reshape(-1,V), tgt.reshape(-1)).exp().item()
        acc = (logits.argmax(-1) == tgt).float().mean().item()
        # Per-half accuracy
        mid = (SEQ-1) // 2
        acc_first = (logits[:,:mid].argmax(-1) == tgt[:,:mid]).float().mean().item()
        acc_second = (logits[:,mid:].argmax(-1) == tgt[:,mid:]).float().mean().item()
    print(f"  {name:16s}: PPL={ppl:.2f} acc={acc*100:.1f}% "
          f"[copy={acc_first*100:.0f}% sum={acc_second*100:.0f}%] "
          f"p={np_:,} t={elapsed:.0f}s")
