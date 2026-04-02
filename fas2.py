"""FAST arch search v2: precomputed features, batch scoring."""
import torch, torch.nn as nn, torch.nn.functional as F
import random as rng; from collections import defaultdict; import time
DEVICE = torch.device('cuda')

class MM(nn.Module):
    def __init__(s,d):super().__init__();s.W=nn.Parameter(torch.randn(d,d)*0.02)
    def forward(s,x):return x@s.W
class Gate(nn.Module):
    def __init__(s,d):super().__init__();s.W=nn.Linear(d,d)
    def forward(s,x):return torch.sigmoid(s.W(x))*x
class Norm(nn.Module):
    def __init__(s,d):super().__init__();s.n=nn.LayerNorm(d)
    def forward(s,x):return s.n(x)
class Act(nn.Module):
    def forward(s,x):return F.gelu(x)
OPS={'mm':MM,'gate':Gate,'norm':Norm,'act':Act}

class GNet(nn.Module):
    def __init__(s,d,genome):
        super().__init__()
        s.ops=nn.ModuleList()
        s.conn=[]
        for op,inp in genome:
            s.ops.append(OPS[op](d) if op!='act' else Act())
            s.conn.append(inp)
    def forward(s,x):
        nodes=[x]
        for i,(op,inp) in enumerate(zip(s.ops,s.conn)):
            nodes.append(op(nodes[inp])+nodes[inp])
        return nodes[-1]

def parse_g6(line):
    data=[ord(c)-63 for c in line.strip()]
    if data[0]<=62:n=data[0];bs=1
    else:n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63);bs=4
    A=torch.zeros(n,n,device=DEVICE);bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6;bw=5-(bi%6)
            if bp<len(data)and(data[bp]>>bw)&1:A[i,j]=A[j,i]=1
            bi+=1
    return A

print("Precomputing features..."); t0=time.time()
D=32; NS=200; all_g=[]
with open(r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6") as f:
    for line in f:
        if line.strip():all_g.append(parse_g6(line))
rng.seed(42); si=rng.sample(range(len(all_g)),NS)
feats=[]
for idx in si:
    adj=all_g[idx];N=adj.shape[0];deg=adj.sum(1).float()
    f=torch.zeros(D,device=DEVICE);sd=deg.sort().values
    f[0]=deg.mean()/4;f[1]=deg.std()/2
    f[2:2+min(N,D-2)]=sd[:D-2]/4
    # Add walk_weight_k4 features
    A4=torch.matrix_power(adj.float(),4)
    ww=A4.sum(1).sort().values; f[10:10+min(N,D-10)]=ww[:D-10]/100
    # Add eigenvalue features
    try:
        eigs=torch.linalg.eigvalsh(adj.float()).sort().values
        f[20:20+min(N,D-20)]=eigs[:D-20]/5
    except:pass
    feats.append(f)
FT=torch.stack(feats)
print(f"  {NS} graphs in {time.time()-t0:.1f}s")

def sc(model):
    model.eval()
    with torch.no_grad():
        e=model(FT);ks=[tuple(e[i].cpu().numpy().round(3)) for i in range(NS)]
    return len(set(ks))

def rg(n=6):
    g=[]
    for i in range(n):g.append((rng.choice(list(OPS.keys())),rng.randint(0,i) if i>0 else 0))
    return g
def mt(g):
    g=list(g);i=rng.randint(0,len(g)-1);op,inp=g[i]
    if rng.random()<0.5:op=rng.choice(list(OPS.keys()))
    else:inp=rng.randint(0,i) if i>0 else 0
    g[i]=(op,inp);return g

POP=20;GENS=15;ND=8
print(f"\nSEARCH: pop={POP} gens={GENS} nodes={ND}")
torch.manual_seed(42);base=GNet(D,[('mm',0),('act',1),('mm',2),('act',3)]).to(DEVICE)
bs=sc(base);print(f"  Baseline: {bs}/{NS}\n")
pop=[rg(ND) for _ in range(POP)];best_s=0;best_g=None;t0=time.time()
for gen in range(GENS):
    res=[]
    for g in pop:
        torch.manual_seed(42);m=GNet(D,g).to(DEVICE);res.append((g,sc(m)))
    res.sort(key=lambda x:-x[1]);gb=res[0]
    if gb[1]>best_s:best_s=gb[1];best_g=gb[0];print(f"  Gen {gen+1:2d}: BEST {best_s}/{NS} {best_g}")
    elif(gen+1)%5==0:print(f"  Gen {gen+1:2d}: {gb[1]}/{NS}")
    elite=[r[0] for r in res[:POP//3]];new=list(elite)
    while len(new)<POP:new.append(mt(rng.choice(elite)))
    pop=new
print(f"\nBaseline:{bs} Evolved:{best_s} Time:{time.time()-t0:.1f}s")
print(f"Genome: {best_g}")
if best_s>bs:print(f"*** +{best_s-bs} IMPROVEMENT ***")
