"""
WAVELET ARCHITECTURE: Multi-resolution in ONE operation.

FFT: global transform (all frequencies at all positions).
Wavelet: LOCAL high-freq + GLOBAL low-freq simultaneously.

This captures both NEARBY patterns (high freq, small window)
and DISTANT patterns (low freq, large window) in ONE pass.

FFT won because it's the natural basis. Wavelet = ADAPTIVE natural basis.
If wavelet > FFT: multi-resolution is the key advantage.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, math
DEVICE = torch.device('cuda')

class WaveletLayer(nn.Module):
    """Simplified multi-resolution: process at 3 scales simultaneously."""
    def __init__(self, d, N):
        super().__init__()
        # Scale 1: every token (local, high freq)
        self.local = nn.Sequential(nn.Linear(d, d), nn.GELU())
        # Scale 2: every 2nd token (medium)
        self.medium = nn.Sequential(nn.Linear(d, d), nn.GELU())
        # Scale 3: every 4th token (global, low freq)
        self.global_ = nn.Sequential(nn.Linear(d, d), nn.GELU())
        # Combine scales
        self.combine = nn.Linear(d*3, d)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape
        # Scale 1: full resolution
        s1 = self.local(x)  # [B, N, D]

        # Scale 2: downsample by 2, process, upsample
        x2 = x[:, ::2]  # [B, N//2, D]
        s2 = self.medium(x2)
        s2 = s2.repeat_interleave(2, dim=1)[:, :N]  # upsample back

        # Scale 3: downsample by 4, process, upsample
        x4 = x[:, ::4]  # [B, N//4, D]
        s3 = self.global_(x4)
        s3 = s3.repeat_interleave(4, dim=1)[:, :N]

        # Combine all scales
        combined = self.combine(torch.cat([s1, s2, s3], dim=-1))
        x = self.norm(x + combined)
        return self.norm2(x + self.ffn(x))

class WaveletModel(nn.Module):
    def __init__(self, V, d, N, nl):
        super().__init__()
        self.embed=nn.Embedding(V,d);self.pos=nn.Embedding(N+1,d)
        self.layers=nn.ModuleList([WaveletLayer(d, N) for _ in range(nl)])
        self.head=nn.Linear(d,V)
    def forward(self, ids):
        B,N=ids.shape;x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers:x=l(x)
        return self.head(x)

class WaveCausal(nn.Module):
    def __init__(s,V,d,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList()
        for _ in range(nl):
            s.layers.append(nn.ModuleDict({
                'freq':nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d)),
                'n1':nn.LayerNorm(d),'ffn':nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),'n2':nn.LayerNorm(d)}))
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        for l in s.layers:
            xf=torch.fft.rfft(x,dim=1);F_len=xf.shape[1];keep=max(F_len//2,1)
            xf_c=xf.clone();xf_c[:,keep:]=0
            xfp=torch.complex(l['freq'](xf_c.real),l['freq'](xf_c.imag))
            xb=torch.fft.irfft(xfp,n=N,dim=1);x=l['n1'](x+xb);x=l['n2'](x+l['ffn'](x))
        return s.head(x)

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
print("WAVELET vs WAVE-FFT vs TRANSFORMER")
print("="*50)

for name,make in [
    ("Transformer",lambda:TF(V,D,4,3,SEQ)),
    ("Wave CAUSAL",lambda:WaveCausal(V,D,3,SEQ)),
    ("Wavelet",lambda:WaveletModel(V,D,N,3)),
]:
    torch.manual_seed(42);model=make().to(DEVICE)
    np_=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(EP):
        inp,tgt=gen(V,SEQ,48)
        loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        inp,tgt=gen(V,SEQ,1000)
        logits=model(inp);ppl=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).exp().item()
        acc=(logits.argmax(-1)==tgt).float().mean().item()
    print(f"  {name:16s}: PPL={ppl:.2f} acc={acc*100:.1f}% p={np_:,}")
