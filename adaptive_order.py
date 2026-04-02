"""
ADAPTIVE ORDER: Connection complexity adapts to task difficulty.

Easy task (linear pattern) -> pair only (cheap, like attention)
Hard task (multi-token pattern) -> pair + triple (expensive but needed)

The ORDER of connection is the key variable, not the connection function.
Attention = fixed order 2. This = learned order 2-3.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class PairConnect(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,d))
    def forward(s,x):
        B,N,D=x.shape
        xi=x.unsqueeze(2).expand(B,N,N,D);xj=x.unsqueeze(1).expand(B,N,N,D)
        p=s.f(torch.cat([xi,xj],-1))
        mask=torch.triu(torch.ones(N,N,device=DEVICE),1).bool()
        p=p.masked_fill(mask.unsqueeze(0).unsqueeze(-1),0)
        w=F.softmax(p.norm(dim=-1).masked_fill(mask.unsqueeze(0),-1e9),-1)
        return (w.unsqueeze(-1)*p).sum(2)

class TripleConnect(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.Linear(d,d))
    def forward(s,x):
        B,N,D=x.shape;S=4
        ji=torch.randint(N,(B,N,S),device=DEVICE);ki=torch.randint(N,(B,N,S),device=DEVICE)
        xj=x.gather(1,ji.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xk=x.gather(1,ki.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        return s.f(torch.cat([x.unsqueeze(2).expand(B,N,S,D),xj,xk],-1)).mean(2)

class AdaptiveModel(nn.Module):
    """ADAPTIVE: learns when to use pair vs triple PER POSITION."""
    def __init__(s,V,d,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        s.layers=nn.ModuleList()
        for _ in range(nl):
            s.layers.append(nn.ModuleDict({
                'pair':PairConnect(d),'triple':TripleConnect(d),
                # PER-POSITION order selector: look at local context to decide
                'order_selector':nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,1),nn.Sigmoid()),
                'n1':nn.LayerNorm(d),
                'ffn':nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                'n2':nn.LayerNorm(d)}))
        s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        for l in s.layers:
            p=l['pair'](x);t=l['triple'](x)
            # Per-position: how much triple vs pair?
            alpha=l['order_selector'](x)  # [B,N,1] in [0,1]
            combined=alpha*t+(1-alpha)*p
            x=l['n1'](x+combined);x=l['n2'](x+l['ffn'](x))
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

# THREE different tasks to test adaptivity
def gen_easy(V=20,seq=20,batch=48):
    """Easy: increment. Only needs 1-step lookback."""
    t=torch.randint(0,V,(batch,seq),device=DEVICE)
    for i in range(1,seq):t[:,i]=(t[:,i-1]+1)%V
    return t[:,:-1],t[:,1:]

def gen_medium(V=20,seq=20,batch=48):
    """Medium: copy 2 back. Needs pair connection."""
    t=torch.randint(0,V,(batch,seq),device=DEVICE)
    for i in range(2,seq):t[:,i]=t[:,i-2]
    return t[:,:-1],t[:,1:]

def gen_hard(V=20,seq=20,batch=48):
    """Hard: fibonacci-like. Needs triple connection."""
    t=torch.randint(0,V,(batch,seq),device=DEVICE)
    for i in range(3,seq):t[:,i]=(t[:,i-1]+t[:,i-2]+t[:,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;EP=120;D=40
print("ADAPTIVE ORDER CONNECTION vs TRANSFORMER")
print("="*55)

for task_name, gen_fn in [("easy(inc)",gen_easy),("medium(copy2)",gen_medium),("hard(fib3)",gen_hard)]:
    print(f"\n--- Task: {task_name} ---")
    for name, make in [
        ("Transformer", lambda: TF(V,D,4,2,SEQ)),
        ("Adaptive", lambda: AdaptiveModel(V,D,2,SEQ)),
    ]:
        torch.manual_seed(42);model=make().to(DEVICE)
        np_=sum(p.numel() for p in model.parameters())
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        for ep in range(EP):
            inp,tgt=gen_fn(V,SEQ,48)
            loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            inp,tgt=gen_fn(V,SEQ,1000)
            acc=(model(inp).argmax(-1)==tgt).float().mean().item()
        print(f"  {name:12s}: acc={acc*100:.1f}% p={np_:,}")
