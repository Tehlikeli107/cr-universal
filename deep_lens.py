"""
DEEP LENS: 3 levels of graph neural network power.
L1: GIN (1-WL) — degree features only
L2: CR-GNN — CR + walk + degree
L3: DEEP — Laplacian PE + Random Walk PE + CR + walk + degree
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from itertools import combinations
from collections import defaultdict
import time, random

DEVICE = torch.device('cuda')

def degree_feat(adj):
    d=adj.sum(1).float(); return torch.stack([d,d**2,(adj@adj).diagonal().float()],1)

def lap_pe(adj, k=8):
    N=adj.shape[0]; L=torch.diag(adj.sum(1).float())-adj.float()
    try:
        ev,evec=torch.linalg.eigh(L); pe=evec[:,1:min(k+1,N)]
        if pe.shape[1]<k: pe=F.pad(pe,(0,k-pe.shape[1]))
        for c in range(pe.shape[1]):
            idx=(pe[:,c].abs()>1e-6).nonzero(as_tuple=True)[0]
            if len(idx)>0 and pe[idx[0],c]<0: pe[:,c]=-pe[:,c]
        return pe
    except: return torch.zeros(N,k,device=DEVICE)

def rw_pe(adj, steps=8):
    N=adj.shape[0]; d=adj.sum(1).float(); di=1.0/(d+1e-10)
    P=(adj.float()*di.unsqueeze(0)).T; pe=torch.zeros(N,steps,device=DEVICE)
    Pk=P.clone()
    for s in range(steps): pe[:,s]=Pk.diagonal(); Pk=Pk@P
    return pe

def cr_feat(adj):
    N=adj.shape[0]; f=torch.zeros(N,4,device=DEVICE)
    for v in range(N):
        nb=[v]+[u for u in range(N) if adj[v,u]]
        if len(nb)<3: continue
        for a,b,c in combinations(nb,3):
            e=int(adj[a,b])+int(adj[a,c])+int(adj[b,c]); f[v,e]+=1
    t=f.sum(1,keepdim=True); return f/(t+1e-10)

def ww_feat(adj, k=4):
    N=adj.shape[0]; Ak=torch.matrix_power(adj.float(),k)
    f=torch.zeros(N,8,device=DEVICE)
    for v in range(N): sv=Ak[v].sort(descending=True).values[:8]; f[v,:len(sv)]=sv
    return f

class GINLayer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))
        self.eps=nn.Parameter(torch.tensor(0.0))
    def forward(self,x,a): return self.mlp((1+self.eps)*x+a.float()@x)

class L1_GIN(nn.Module):
    def __init__(self,d=64,nl=3):
        super().__init__()
        self.e=nn.Linear(3,d); self.ls=nn.ModuleList([GINLayer(d) for _ in range(nl)])
    def forward(self,a):
        x=self.e(degree_feat(a))
        for l in self.ls: x=l(x,a)
        return x.sum(0)

class L2_CRGNN(nn.Module):
    def __init__(self,d=64,nl=3):
        super().__init__()
        self.e=nn.Linear(15,d); self.ls=nn.ModuleList([GINLayer(d) for _ in range(nl)])
    def forward(self,a):
        x=self.e(torch.cat([cr_feat(a),ww_feat(a),degree_feat(a)],1))
        for l in self.ls: x=l(x,a)
        return x.sum(0)

class L3_Deep(nn.Module):
    def __init__(self,d=64,nl=3):
        super().__init__()
        self.e=nn.Linear(31,d); self.ls=nn.ModuleList([GINLayer(d) for _ in range(nl)])
    def forward(self,a):
        x=self.e(torch.cat([lap_pe(a,8),rw_pe(a,8),cr_feat(a),ww_feat(a),degree_feat(a)],1))
        for l in self.ls: x=l(x,a)
        return x.sum(0)

def parse_g6(line):
    line=line.strip(); data=[ord(c)-63 for c in line]
    if data[0]<=62: n=data[0];bs=1
    else: n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63);bs=4
    A=torch.zeros(n,n,device=DEVICE); bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6;bw=5-(bi%6)
            if bp<len(data) and (data[bp]>>bw)&1: A[i,j]=A[j,i]=1
            bi+=1
    return A

def test_pow(model, graphs):
    model.eval(); embs=defaultdict(list)
    with torch.no_grad():
        for i,a in enumerate(graphs):
            e=model(a); k=tuple(e.cpu().numpy().round(3)); embs[k].append(i)
    return len(embs), sum(1 for v in embs.values() if len(v)>1)

if __name__=="__main__":
    print("3-LEVEL LENS: GIN vs CR-GNN vs DEEP LENS")
    print("="*50)
    all_g=[]
    with open(r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6") as f:
        for line in f:
            if line.strip(): all_g.append(parse_g6(line))
    random.seed(42); S=500
    graphs=[all_g[i] for i in random.sample(range(len(all_g)),S)]
    print(f"Testing {S} of {len(all_g)} n=8 graphs\n")
    for name,M in [("L1:GIN",L1_GIN),("L2:CR-GNN",L2_CRGNN),("L3:DEEP",L3_Deep)]:
        torch.manual_seed(42); m=M(64,3).to(DEVICE)
        np_=sum(p.numel() for p in m.parameters())
        t0=time.time(); nu,nc=test_pow(m,graphs); t1=time.time()-t0
        print(f"  {name:12s}: {nu}/{S} ({nu/S*100:.1f}%) coll={nc} p={np_} t={t1:.0f}s")
