"""FAIR: Space Bender vs Transformer at same param count."""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class SpaceBendLayer(nn.Module):
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
        # Causal
        for o in range(1,s.w+1):feats[:,:,D*(s.w+o):D*(s.w+o+1)]=0
        mixed=s.local_mix(feats);out=s.unwarp(mixed)
        x=s.norm(x+out);return s.norm2(x+s.ffn(x))

class SB(nn.Module):
    def __init__(s,V,d,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList([SpaceBendLayer(d,w=2+i) for i in range(nl)])
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        for l in s.layers:x=l(x)
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

V=20;SEQ=20;EP=150
print("FAIR: Space Bender vs Transformer (matched params)")
print("="*55)

configs=[
    ("TF d=40 3L",lambda:TF(V,40,4,3,SEQ)),
    ("TF d=48 3L",lambda:TF(V,48,4,3,SEQ)),
    ("TF d=64 3L",lambda:TF(V,64,4,3,SEQ)),
    ("SB d=32 3L",lambda:SB(V,32,3,SEQ)),
    ("SB d=40 3L",lambda:SB(V,40,3,SEQ)),
    ("SB d=48 3L",lambda:SB(V,48,3,SEQ)),
]

results=[]
for name,make in configs:
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
    results.append((name,np_,ppl,acc))
    print(f"  {name:14s}: p={np_:6,} PPL={ppl:.2f} acc={acc*100:.1f}%")

print(f"\nSorted by params:")
for n,p,ppl,a in sorted(results,key=lambda x:x[1]):
    print(f"  {n:14s}: {p:6,} params PPL={ppl:.2f} acc={a*100:.1f}%")
