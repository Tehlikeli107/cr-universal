"""FAIR comparison: same parameter count. Hybrid d=34 vs Transformer d=48."""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class LPair(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,d))
    def forward(s,x):
        B,N,D=x.shape
        xi=x.unsqueeze(2).expand(B,N,N,D);xj=x.unsqueeze(1).expand(B,N,N,D)
        p=s.f(torch.cat([xi,xj],-1))
        mask=torch.triu(torch.ones(N,N,device=DEVICE),1).bool()
        p=p.masked_fill(mask.unsqueeze(0).unsqueeze(-1),0)
        w=p.norm(dim=-1,keepdim=True);w=w.masked_fill(mask.unsqueeze(0).unsqueeze(-1),-1e9)
        w=F.softmax(w.squeeze(-1),-1).unsqueeze(-1)
        return (w*p).sum(2)

class LTriple(nn.Module):
    def __init__(s,d,ns=4):
        super().__init__()
        s.ns=ns;s.f=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.Linear(d,d))
    def forward(s,x):
        B,N,D=x.shape;S=s.ns
        ji=torch.randint(N,(B,N,S),device=DEVICE);ki=torch.randint(N,(B,N,S),device=DEVICE)
        xj=x.gather(1,ji.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xk=x.gather(1,ki.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xi=x.unsqueeze(2).expand(B,N,S,D)
        return s.f(torch.cat([xi,xj,xk],-1)).mean(2)

class Hybrid(nn.Module):
    def __init__(s,V,d,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList()
        for _ in range(nl):
            s.layers.append(nn.ModuleDict({
                'pair':LPair(d),'triple':LTriple(d,4),
                'gate':nn.Linear(d*2,d),'n1':nn.LayerNorm(d),
                'ffn':nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                'n2':nn.LayerNorm(d)}))
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        for l in s.layers:
            p=l['pair'](x);t=l['triple'](x)
            g=torch.sigmoid(l['gate'](torch.cat([p,t],-1)))
            x=l['n1'](x+g*p+(1-g)*t);x=l['n2'](x+l['ffn'](x))
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
print("FAIR COMPARISON: Same ~60K params")
print("="*50)

configs = [
    ("Transformer d=48", lambda: TF(V,48,4,2,SEQ)),
    ("Transformer d=64", lambda: TF(V,64,4,2,SEQ)),
    ("Hybrid d=32", lambda: Hybrid(V,32,2,SEQ)),
    ("Hybrid d=36", lambda: Hybrid(V,36,2,SEQ)),
    ("Hybrid d=28", lambda: Hybrid(V,28,2,SEQ)),
]

results = []
for name, make in configs:
    torch.manual_seed(42); model=make().to(DEVICE)
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
        mid=(SEQ-1)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    results.append((name,ppl,acc,a1,a2,np_,elapsed))
    print(f"  {name:18s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")

print(f"\nPARAM-MATCHED COMPARISON:")
# Find closest param counts
for r in sorted(results, key=lambda x:x[5]):
    print(f"  {r[0]:18s}: {r[5]:6,} params -> PPL={r[1]:.2f} acc={r[2]*100:.1f}%")
