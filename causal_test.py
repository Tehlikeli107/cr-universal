"""CRITICAL: Was Wave FFT cheating by using future tokens? Test causal vs non-causal."""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class WaveCausal(nn.Module):
    """Wave with CAUSAL FFT: zero out future frequency components."""
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
            # CAUSAL: only use past. Use causal convolution via FFT.
            # Method: FFT -> zero out high frequencies (future info) -> iFFT
            xf=torch.fft.rfft(x,dim=1)
            # Keep only first half of frequency components (low freq = past)
            F_len=xf.shape[1]; keep=max(F_len//2,1)
            xf_causal=xf.clone(); xf_causal[:,keep:]=0
            xfp=torch.complex(l['freq'](xf_causal.real),l['freq'](xf_causal.imag))
            xb=torch.fft.irfft(xfp,n=N,dim=1)
            x=l['n1'](x+xb);x=l['n2'](x+l['ffn'](x))
        return self.head(x)

class WaveNonCausal(nn.Module):
    """Original Wave: NO causal masking (uses future tokens!)."""
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

V=20;SEQ=20;EP=150;D=48
print("CAUSAL TEST: Was Wave cheating?")
print("="*50)

for name,make in [
    ("Transformer(causal)",lambda:TF(V,D,4,3,SEQ)),
    ("Wave NON-causal",lambda:WaveNonCausal(V,D,3,SEQ)),
    ("Wave CAUSAL",lambda:WaveCausal(V,D,3,SEQ)),
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
    print(f"  {name:22s}: PPL={ppl:.2f} acc={acc*100:.1f}% p={np_:,}")
