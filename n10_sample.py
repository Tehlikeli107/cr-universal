"""n=10 test on 1M sample (from 12M total). If complete on sample -> likely complete on all."""
import numpy as np
from collections import defaultdict
import time

def parse_g6(line):
    line=line.strip(); data=[ord(c)-63 for c in line]
    if data[0]<=62: n=data[0];bs=1
    else: n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63);bs=4
    A=np.zeros((n,n),dtype=np.int8); bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6;bw=5-(bi%6)
            if bp<len(data) and (data[bp]>>bw)&1: A[i,j]=A[j,i]=1
            bi+=1
    return A

def deg(A): return tuple(sorted(np.sum(A,1)))
def ww4(A): A4=np.linalg.matrix_power(A.astype(int),4); return tuple(sorted([tuple(sorted(A4[i])) for i in range(10)]))
def eig(A): return tuple(sorted(np.round(np.linalg.eigvalsh(A.astype(float)),4)))

LIMIT = 1000000
print(f"n=10 COMPLETENESS TEST (first {LIMIT:,} of 12M graphs)")
print("="*55)

# Phase 1: Load + degree group
print("Phase 1: Loading + degree grouping...")
t0=time.time()
deg_groups=defaultdict(list); graphs={}; n=0
with open(r"C:\Users\salih\Desktop\cr-universal\graph10_raw.g6") as f:
    for line in f:
        if n>=LIMIT: break
        line=line.strip()
        if not line: continue
        A=parse_g6(line); d=deg(A)
        if len(deg_groups[d])>0 or True:  # store all for now
            graphs[n]=A
        deg_groups[d].append(n); n+=1
        if n%200000==0: print(f"  {n:,}...",flush=True)
print(f"  {n:,} graphs, {len(deg_groups):,} degree seqs, {time.time()-t0:.0f}s")

# Collision groups
coll={k:v for k,v in deg_groups.items() if len(v)>1}
nc=sum(len(v) for v in coll.values())
print(f"  Degree collisions: {len(coll):,} groups ({nc:,} graphs)")

# Phase 2: walk_weight on collision groups
print(f"\nPhase 2: walk_weight_k4 on {nc:,} graphs...")
t0=time.time()
ww_groups=defaultdict(list); done=0
for dk,indices in coll.items():
    for i in indices:
        w=ww4(graphs[i]); ww_groups[(dk,w)].append(i)
        done+=1
        if done%20000==0: print(f"  {done:,}/{nc:,}...",flush=True)
ww_coll={k:v for k,v in ww_groups.items() if len(v)>1}
nwc=len(ww_coll); nwg=sum(len(v) for v in ww_coll.values())
print(f"  After walk_weight: {nwc:,} collision groups ({nwg:,} graphs), {time.time()-t0:.0f}s")

# Phase 3: eigenvalues
if nwc>0:
    print(f"\nPhase 3: eigenvalues on {nwg:,} graphs...")
    t0=time.time()
    eig_groups=defaultdict(list)
    for k,indices in ww_coll.items():
        for i in indices:
            e=eig(graphs[i]); eig_groups[(k,e)].append(i)
    eig_coll={k:v for k,v in eig_groups.items() if len(v)>1}
    nec=len(eig_coll)
    print(f"  After eigenvalues: {nec} collision groups, {time.time()-t0:.0f}s")
else: nec=0

# Phase 4: betweenness
if nec>0:
    import networkx as nx
    print(f"\nPhase 4: betweenness on {nec} groups...")
    final=0
    for k,indices in eig_coll.items():
        bc_map=defaultdict(list)
        for i in indices:
            G=nx.Graph(graphs[i])
            bc=tuple(sorted([round(v,4) for v in nx.betweenness_centrality(G).values()]))
            bc_map[bc].append(i)
        for v in bc_map.values():
            if len(v)>1: final+=1; print(f"  HARD: {v}")
    nec=final
    print(f"  After betweenness: {final} collision groups")
else: final=0

print(f"\n{'='*55}")
print(f"n=10 RESULT ({n:,} graphs sampled):")
print(f"  degree: {len(coll):,} collisions")
print(f"  +walk_weight: {len(ww_coll):,}")
print(f"  +eigenvalue: {nec}")
print(f"  +betweenness: {final}")
if final==0:
    print(f"\n*** COMPLETE on {n:,} graph sample! ***")
