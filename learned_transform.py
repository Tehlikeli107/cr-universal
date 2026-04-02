"""
LEARNED TRANSFORM: Instead of fixed FFT (sin/cos basis),
learn the BEST basis functions for the task.

FFT wins because sin/cos is the NATURAL basis for copy/sum tasks.
But for OTHER tasks, a DIFFERENT basis might be better.

Architecture:
1. Learn a transform matrix T (replaces FFT)
2. Forward: T @ x (transform to learned domain)
3. Process in learned domain
4. Inverse: T^{-1} @ processed (back to token domain)

T starts random. Training finds the OPTIMAL basis.
If T converges to DFT matrix = FFT was already optimal.
If T converges to something ELSE = we found a BETTER basis.

Also test: FFT + learned pair attention IN FREQUENCY DOMAIN.
This combines Pattern 1 (natural space) + Pattern 2 (selectivity).
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, math
DEVICE = torch.device('cuda')

class LearnedTransformLayer(nn.Module):
    """Learned basis transform + processing + inverse transform."""
    def __init__(self, d, N):
        super().__init__()
        # Learnable transform matrix (N x N, operates on sequence dim)
        self.T = nn.Parameter(torch.randn(N, N) * 0.1)
        # Process in transformed domain
        self.process = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape
        # Causal: zero out upper triangle of T (can only use past info)
        T_causal = self.T.tril()

        # Transform: [B, N, D] -> T @ [B, N, D] along seq dim
        x_transformed = torch.matmul(T_causal.unsqueeze(0), x)  # [B, N, D]

        # Process in transformed space
        x_processed = self.process(x_transformed)

        # Inverse transform (use transpose for approximate inverse)
        x_back = torch.matmul(T_causal.T.unsqueeze(0), x_processed)

        x = self.norm(x + x_back)
        return self.norm2(x + self.ffn(x))


class FreqAttentionLayer(nn.Module):
    """FFT -> attention IN frequency domain -> iFFT.
    Combines natural space (Pattern 1) + selectivity (Pattern 2)."""
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.d = d; self.nh = n_heads; self.dh = d // n_heads
        # Attention in frequency domain
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape
        # FFT along sequence
        x_freq = torch.fft.rfft(x, dim=1)  # [B, N//2+1, D] complex
        x_freq_real = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # [B, F, 2D]

        # Pad back to match expected dims
        F_len = x_freq_real.shape[1]
        # Simple: just do attention on frequency components
        Q = self.Wq(x_freq.real); K = self.Wk(x_freq.real); V = self.Wv(x_freq.real)
        # Standard attention in frequency domain
        attn = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(self.dh), dim=-1)
        freq_out = attn @ V
        freq_out = self.Wo(freq_out)

        # iFFT back
        x_freq_new = torch.complex(freq_out, x_freq.imag)
        x_back = torch.fft.irfft(x_freq_new, n=N, dim=1)

        x = self.norm(x + x_back)
        return self.norm2(x + self.ffn(x))


class LearnedTransformModel(nn.Module):
    def __init__(self, V, d, N, nl):
        super().__init__()
        self.embed=nn.Embedding(V,d); self.pos=nn.Embedding(N+1,d)
        self.layers = nn.ModuleList([LearnedTransformLayer(d, N) for _ in range(nl)])
        self.head = nn.Linear(d, V)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers: x=l(x)
        return self.head(x)

class FreqAttnModel(nn.Module):
    def __init__(self, V, d, nl, maxN):
        super().__init__()
        self.embed=nn.Embedding(V,d); self.pos=nn.Embedding(maxN,d)
        self.layers = nn.ModuleList([FreqAttentionLayer(d) for _ in range(nl)])
        self.head = nn.Linear(d, V)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers: x=l(x)
        return self.head(x)

class WaveModel(nn.Module):
    def __init__(self, V, d, nl, maxN):
        super().__init__()
        self.embed=nn.Embedding(V,d);self.pos=nn.Embedding(maxN,d)
        self.layers=nn.ModuleList()
        for _ in range(nl):
            self.layers.append(nn.ModuleDict({
                'freq':nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d)),
                'n1':nn.LayerNorm(d),
                'ffn':nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                'n2':nn.LayerNorm(d)}))
        self.head=nn.Linear(d,V)
    def forward(self,ids):
        B,N=ids.shape;x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers:
            xf=torch.fft.rfft(x,dim=1)
            xfp=torch.complex(l['freq'](xf.real),l['freq'](xf.imag))
            xb=torch.fft.irfft(xfp,n=N,dim=1)
            x=l['n1'](x+xb);x=l['n2'](x+l['ffn'](x))
        return self.head(x)

class TF(nn.Module):
    def __init__(s,V,d,nh,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl);s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        return s.head(s.enc(x,mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()))

def gen(V=20,seq=20,batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE);mid=seq//2
    for b in range(batch):
        for i in range(2,mid):t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq):t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;N=SEQ-1;EP=150;D=48
print("LEARNED TRANSFORM vs WAVE vs FREQ-ATTENTION vs TRANSFORMER")
print("="*60)

for name,make in [
    ("Transformer",lambda:TF(V,D,4,3,SEQ)),
    ("Wave (FFT)",lambda:WaveModel(V,D,3,SEQ)),
    ("LearnedBasis",lambda:LearnedTransformModel(V,D,N,3)),
    ("FreqAttention",lambda:FreqAttnModel(V,D,3,SEQ)),
]:
    torch.manual_seed(42);model=make().to(DEVICE)
    np_=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    t0=time.time()
    for ep in range(EP):
        inp,tgt=gen(V,SEQ,48)
        loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    elapsed=time.time()-t0
    with torch.no_grad():
        inp,tgt=gen(V,SEQ,1000)
        logits=model(inp);ppl=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).exp().item()
        acc=(logits.argmax(-1)==tgt).float().mean().item()
        mid=N//2;a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:16s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
