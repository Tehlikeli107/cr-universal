"""FAST ARCH SEARCH: zero-shot eval with random weights. No training needed."""
import torch, torch.nn as nn, torch.nn.functional as F
import random as rng
from collections import defaultdict
import time
DEVICE = torch.device('cuda')

class MatMul(nn.Module):
    def __init__(self,d): super().__init__(); self.W=nn.Parameter(torch.randn(d,d)*0.02)
    def forward(self,x): return x@self.W
class Gate(nn.Module):
    def __init__(self,d): super().__init__(); self.W=nn.Linear(d,d)
    def forward(self,x): return torch.sigmoid(self.W(x))*x
class Norm(nn.Module):
    def __init__(self,d): super().__init__(); self.n=nn.LayerNorm(d)
    def forward(self,x): return self.n(x)
class Act(nn.Module):
    def forward(self,x): return F.gelu(x)
class Scale(nn.Module):
    def __init__(self,d): super().__init__(); self.s=nn.Parameter(torch.ones(d))
    def forward(self,x): return x*self.s

OPS={'mm':MatMul,'gate':Gate,'norm':Norm,'act':Act,'scale':Scale}

class GenomeNet(nn.Module):
    def __init__(self,d,genome):
        super().__init__()
        self.ops=nn.ModuleList()
        self.conn=[]
        for op,inp in genome:
            self.ops.append(OPS[op](d) if op in ['mm','gate','norm','scale'] else Act())
            self.conn.append(inp)
    def forward(self,x):
        nodes=[x.sum(dim=1)]
        for i,(op,inp) in enumerate(zip(self.ops,self.conn)):
            nodes.append(op(nodes[inp])+nodes[inp])
        return nodes[-1]

def parse_g6(line):
    line=line.strip();data=[ord(c)-63 for c in line]
    if data[0]<=62:n=data[0];bs=1
    else:n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63);bs=4
    A=torch.zeros(n,n,device=DEVICE);bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6;bw=5-(bi%6)
            if bp<len(data)and(data[bp]>>bw)&1:A[i,j]=A[j,i]=1
            bi+=1
    return A

def g2feat(adj,d=32):
    N=adj.shape[0];deg=adj.sum(1).float()
    try:eig=torch.linalg.eigvalsh(adj.float())
    except:eig=torch.zeros(N,device=DEVICE)
    f=torch.zeros(N,d,device=DEVICE);f[:,0]=deg/4
    f[:,1:min(N+1,d)]=eig.unsqueeze(0).expand(N,-1)[:,:d-1]/5
    return f.unsqueeze(0)

def score(model,graphs,d=32):
    model.eval();embs=defaultdict(list)
    with torch.no_grad():
        for i,adj in enumerate(graphs):
            e=model(g2feat(adj,d));k=tuple(e[0].cpu().numpy().round(3));embs[k].append(i)
    return len(embs)

def rand_genome(n=6):
    g=[]
    for i in range(n):
        g.append((rng.choice(list(OPS.keys())),rng.randint(0,i) if i>0 else 0))
    return g

def mutate(g):
    g=list(g);i=rng.randint(0,len(g)-1);op,inp=g[i]
    if rng.random()<0.5:op=rng.choice(list(OPS.keys()))
    else:inp=rng.randint(0,i) if i>0 else 0
    g[i]=(op,inp);return g

g8=r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
all_g=[];
with open(g8) as f:
    for line in f:
        if line.strip():all_g.append(parse_g6(line))
rng.seed(42);sample=[all_g[i] for i in rng.sample(range(len(all_g)),200)]
D=32;POP=15;GENS=10;NODES=6
print(f"FAST ARCH SEARCH: {POP} pop, {GENS} gens, {NODES} nodes, 500 graphs")
print("="*55)

# Baseline
torch.manual_seed(42)
base=GenomeNet(D,[('mm',0),('act',1),('mm',2),('act',3)]).to(DEVICE)
ns=score(base,sample,D);print(f"  Baseline (mm chain): {ns}/200\n")

pop=[rand_genome(NODES) for _ in range(POP)];best_s=0;best_g=None
t0=time.time()
for gen in range(GENS):
    res=[]
    for g in pop:
        torch.manual_seed(42);m=GenomeNet(D,g).to(DEVICE)
        s=score(m,sample,D);res.append((g,s))
    res.sort(key=lambda x:-x[1]);gb=res[0]
    if gb[1]>best_s:best_s=gb[1];best_g=gb[0];print(f"  Gen {gen+1:2d}: NEW BEST {best_s}/200 genome={best_g[:3]}...")
    elif(gen+1)%5==0:print(f"  Gen {gen+1:2d}: {gb[1]}/200")
    elite=[r[0] for r in res[:POP//3]];new=list(elite)
    while len(new)<POP:new.append(mutate(rng.choice(elite)))
    pop=new

print(f"\n{'='*55}")
print(f"Baseline: {ns}/200 | Best evolved: {best_s}/200 | Time: {time.time()-t0:.1f}s")
print(f"Best genome: {best_g}")
if best_s>ns:print(f"\n*** MACHINE FOUND +{best_s-ns} BETTER ARCHITECTURE! ***")
