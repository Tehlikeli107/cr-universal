"""Hybrid: LearnedPair + LearnedTriple combined. Best of both."""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class LPair(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d*2,d*2),nn.GELU(),nn.Linear(d*2,d))
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
    def __init__(s,d,ns=6):
        super().__init__()
        s.ns=ns;s.f=nn.Sequential(nn.Linear(d*3,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(s,x):
        B,N,D=x.shape;S=s.ns
        ji=torch.randint(N,(B,N,S),device=DEVICE);ki=torch.randint(N,(B,N,S),device=DEVICE)
        xj=x.gather(1,ji.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xk=x.gather(1,ki.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xi=x.unsqueeze(2).expand(B,N,S,D)
        return s.f(torch.cat([xi,xj,xk],-1)).mean(2)

class HybridModel(nn.Module):
    def __init__(s,V,d=48,nl=2,maxN=24):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList()
        for i in range(nl):
            s.layers.append(nn.ModuleDict({
                'pair':LPair(d),'triple':LTriple(d,6),
                'gate':nn.Linear(d*2,d),
                'n1':nn.LayerNorm(d),
                'ffn':nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                'n2':nn.LayerNorm(d)}))
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        for l in s.layers:
            p=l['pair'](x);t=l['triple'](x)
            # Learned gate: how much pair vs triple
            g=torch.sigmoid(l['gate'](torch.cat([p,t],-1)))
            combined=g*p+(1-g)*t
            x=l['n1'](x+combined);x=l['n2'](x+l['ffn'](x))
        return s.head(x)

class TF(nn.Module):
    def __init__(s,V,d=48,nh=4,nl=2,maxN=24):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl);s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()
        return s.head(s.enc(x,mask=mask))

def gen(V=20,seq=20,batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE);mid=seq//2
    for b in range(batch):
        for i in range(2,mid):t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq):t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;EP=150;D=48
print("HYBRID (Pair+Triple) vs Transformer")
print("="*50)

for name,make in [("Transformer",lambda:TF(V,D,4,2,SEQ)),
                   ("LearnedPair",lambda:learned_connect_make_pair(V,D,SEQ)),
                   ("Hybrid P+T",lambda:HybridModel(V,D,2,SEQ))]:
    if name=="LearnedPair":
        # inline pair-only model
        class PairOnly(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed=nn.Embedding(V,D);s.pos=nn.Embedding(SEQ,D)
                s.layers=nn.ModuleList([nn.ModuleDict({'c':LPair(D),'n1':nn.LayerNorm(D),'ffn':nn.Sequential(nn.Linear(D,D*4),nn.GELU(),nn.Linear(D*4,D)),'n2':nn.LayerNorm(D)}) for _ in range(2)])
                s.head=nn.Linear(D,V)
            def forward(s,ids):
                B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
                for l in s.layers:x=l['n1'](x+l['c'](x));x=l['n2'](x+l['ffn'](x))
                return s.head(x)
        make=lambda:PairOnly()

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
        mid=(SEQ-1)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:14s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
