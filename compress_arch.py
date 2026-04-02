"""
COMPRESSION ARCHITECTURE: Compress N tokens to K, operate, expand back.

N=20 tokens -> compress to K=5 -> process in compressed space -> expand to N=20

Compression FORCES global information capture (must fit N into K).
Processing in K-space is CHEAP (K << N).
Expansion reconstructs per-token predictions.

This is OUTSIDE attention paradigm: no token-token comparison.
Instead: all tokens -> bottleneck -> all tokens.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class CompressLayer(nn.Module):
    """Compress -> Process -> Expand."""
    def __init__(self, d, N, K):
        super().__init__()
        # Compress: N tokens -> K tokens (learned compression)
        self.compress = nn.Linear(N, K)  # operates on sequence dim
        # Process in compressed space
        self.process = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        # Expand: K tokens -> N tokens
        self.expand = nn.Linear(K, N)  # back to sequence dim
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        """x: [B, N, D]"""
        B, N, D = x.shape
        # Compress: [B, N, D] -> [B, D, N] -> compress -> [B, D, K] -> [B, K, D]
        compressed = self.compress(x.transpose(1,2)).transpose(1,2)  # [B, K, D]
        # Process in compressed space
        processed = self.process(compressed)  # [B, K, D]
        # Expand: [B, K, D] -> [B, D, K] -> expand -> [B, D, N] -> [B, N, D]
        expanded = self.expand(processed.transpose(1,2)).transpose(1,2)  # [B, N, D]
        # Residual
        x = self.norm(x + expanded)
        x = self.norm2(x + self.ffn(x))
        return x

class CompressModel(nn.Module):
    def __init__(self, V, d, N, K, nl):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Embedding(N, d)
        # Multiple compress layers with DIFFERENT K (multi-resolution)
        self.layers = nn.ModuleList([
            CompressLayer(d, N, max(K - i, 2)) for i in range(nl)
        ])
        self.head = nn.Linear(d, V)

    def forward(self, ids):
        B, N = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(N, device=ids.device))
        for l in self.layers:
            x = l(x)
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

# Also: WAVE architecture (FFT-based)
class WaveLayer(nn.Module):
    """FFT -> process in frequency domain -> iFFT."""
    def __init__(self, d):
        super().__init__()
        self.freq_process = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape
        # FFT along sequence dimension
        x_freq = torch.fft.rfft(x, dim=1)  # [B, N//2+1, D] complex
        # Process in frequency domain (real part only for simplicity)
        x_freq_real = self.freq_process(x_freq.real)
        x_freq_imag = self.freq_process(x_freq.imag)
        x_freq_proc = torch.complex(x_freq_real, x_freq_imag)
        # iFFT back
        x_time = torch.fft.irfft(x_freq_proc, n=N, dim=1)  # [B, N, D]
        x = self.norm(x + x_time)
        x = self.norm2(x + self.ffn(x))
        return x

class WaveModel(nn.Module):
    def __init__(self, V, d, nl, maxN):
        super().__init__()
        self.embed=nn.Embedding(V,d);self.pos=nn.Embedding(maxN,d)
        self.layers=nn.ModuleList([WaveLayer(d) for _ in range(nl)])
        self.head=nn.Linear(d,V)
    def forward(self, ids):
        B,N=ids.shape;x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers:x=l(x)
        return self.head(x)

def gen(V=20,seq=20,batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE);mid=seq//2
    for b in range(batch):
        for i in range(2,mid):t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq):t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;N=SEQ-1;EP=150;D=48
print("5 PARADIGMS: Transformer vs SpaceBend vs Compress vs Wave")
print("="*55)

# SpaceBend inline (avoid import that runs main)
class SpaceBendLayer2(nn.Module):
    def __init__(s,d,w=3):
        super().__init__()
        s.w=w;s.warp=nn.Linear(d,d,bias=False)
        s.local_mix=nn.Linear(d*(2*w+1),d)
        s.unwarp=nn.Linear(d,d,bias=False)
        s.norm=nn.LayerNorm(d);s.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d));s.norm2=nn.LayerNorm(d)
    def forward(s,x):
        B,N,D=x.shape;xw=s.warp(x)
        padded=F.pad(xw,(0,0,s.w,s.w))
        feats=torch.cat([padded[:,s.w+o:s.w+o+N] for o in range(-s.w,s.w+1)],dim=-1)
        for o in range(1,s.w+1):feats[:,:,D*(s.w+o):D*(s.w+o+1)]=0
        mixed=s.local_mix(feats);out=s.unwarp(mixed)
        x=s.norm(x+out);return s.norm2(x+s.ffn(x))

class SB(nn.Module):
    def __init__(s,V,d,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList([SpaceBendLayer2(d,w=2+i) for i in range(nl)])
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N2=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N2,device=ids.device))
        for l in s.layers:x=l(x)
        return s.head(x)

for name,make in [
    ("Transformer", lambda: TF(V,D,4,3,SEQ)),
    ("SpaceBend", lambda: SB(V,D,3,SEQ)),
    ("Compress K=5", lambda: CompressModel(V,D,N,5,3)),
    ("Compress K=8", lambda: CompressModel(V,D,N,8,3)),
    ("Wave (FFT)", lambda: WaveModel(V,D,3,SEQ)),
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
        mid=(N)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:14s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
