"""
N=10 COMPLETENESS TEST: walk_weight_k4 + eigenvalue + betweenness
on ALL 12,005,168 non-isomorphic graphs.

Strategy (memory efficient):
1. Stream graphs, group by degree sequence (fast, O(N))
2. Within collision groups only: compute walk_weight_k4 (O(N^4) per graph)
3. Remaining collisions: add eigenvalue
4. Still remaining: add betweenness
5. Report: 0 collisions = COMPLETE = historic result
"""
import numpy as np
from collections import defaultdict
import time, gc

def parse_graph6(line):
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

def deg_sig(A):
    return tuple(sorted(np.sum(A,1)))

def ww4_sig(A):
    A4=np.linalg.matrix_power(A.astype(int),4)
    return tuple(sorted([tuple(sorted(A4[i])) for i in range(A.shape[0])]))

def eig_sig(A):
    return tuple(sorted(np.round(np.linalg.eigvalsh(A.astype(float)),4)))

g10_path = r"C:\Users\salih\Desktop\cr-universal\graph10.g6"

# Phase 1: Group by degree sequence (stream, low memory)
print("PHASE 1: Degree sequence grouping (streaming)...")
t0 = time.time()
deg_groups = defaultdict(list)
n_total = 0
with open(g10_path, encoding='latin-1') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        A = parse_graph6(line)
        d = deg_sig(A)
        deg_groups[d].append(n_total)
        n_total += 1
        if n_total % 500000 == 0:
            print(f"  {n_total:,} graphs processed...", flush=True)

n_deg_coll = sum(1 for v in deg_groups.values() if len(v) > 1)
n_deg_coll_graphs = sum(len(v) for v in deg_groups.values() if len(v) > 1)
print(f"  Total: {n_total:,} graphs, {len(deg_groups):,} unique degree seqs")
print(f"  Collision groups: {n_deg_coll:,} ({n_deg_coll_graphs:,} graphs)")
print(f"  Time: {time.time()-t0:.0f}s")

# Phase 2: For collision groups, load graphs and compute walk_weight
print(f"\nPHASE 2: walk_weight_k4 on {n_deg_coll_graphs:,} collision graphs...")
t0 = time.time()

# Need to re-read graphs in collision groups
coll_indices = set()
for indices in deg_groups.values():
    if len(indices) > 1:
        coll_indices.update(indices)

print(f"  Loading {len(coll_indices):,} graphs from collision groups...")
coll_graphs = {}
idx = 0
with open(g10_path, encoding='latin-1') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if idx in coll_indices:
            coll_graphs[idx] = parse_graph6(line)
        idx += 1
        if idx % 500000 == 0:
            print(f"  {idx:,}/{n_total:,} scanned...", flush=True)

print(f"  Loaded {len(coll_graphs):,} graphs")

# Group by degree + walk_weight
ww_groups = defaultdict(list)
processed = 0
for deg_key, indices in deg_groups.items():
    if len(indices) <= 1: continue
    for i in indices:
        A = coll_graphs[i]
        ww = ww4_sig(A)
        ww_groups[(deg_key, ww)].append(i)
        processed += 1
        if processed % 50000 == 0:
            print(f"  walk_weight: {processed:,}/{n_deg_coll_graphs:,}...", flush=True)

ww_coll = {k:v for k,v in ww_groups.items() if len(v) > 1}
n_ww_coll = len(ww_coll)
n_ww_coll_graphs = sum(len(v) for v in ww_coll.values())
print(f"  After walk_weight: {n_ww_coll:,} collision groups ({n_ww_coll_graphs:,} graphs)")
print(f"  Time: {time.time()-t0:.0f}s")

# Phase 3: Add eigenvalues
if n_ww_coll > 0:
    print(f"\nPHASE 3: Adding eigenvalues to {n_ww_coll_graphs:,} remaining graphs...")
    t0 = time.time()
    eig_groups = defaultdict(list)
    for key, indices in ww_coll.items():
        for i in indices:
            A = coll_graphs[i]
            e = eig_sig(A)
            eig_groups[(key, e)].append(i)

    eig_coll = {k:v for k,v in eig_groups.items() if len(v) > 1}
    n_eig_coll = len(eig_coll)
    print(f"  After eigenvalues: {n_eig_coll:,} collision groups")
    print(f"  Time: {time.time()-t0:.0f}s")
else:
    n_eig_coll = 0

# Phase 4: Add betweenness if needed
if n_eig_coll > 0:
    import networkx as nx
    print(f"\nPHASE 4: Adding betweenness to {n_eig_coll} remaining groups...")
    t0 = time.time()
    final_coll = 0
    for key, indices in eig_coll.items():
        bc_map = defaultdict(list)
        for i in indices:
            G = nx.Graph(coll_graphs[i])
            bc = tuple(sorted([round(v,4) for v in nx.betweenness_centrality(G).values()]))
            bc_map[bc].append(i)
        for bc_indices in bc_map.values():
            if len(bc_indices) > 1:
                final_coll += 1
                print(f"  STILL HARD: graphs {bc_indices}")
    print(f"  After betweenness: {final_coll} collision groups")
    print(f"  Time: {time.time()-t0:.0f}s")
else:
    final_coll = 0

# RESULT
print(f"\n{'='*55}")
print(f"N=10 COMPLETENESS RESULT ({n_total:,} graphs)")
print(f"{'='*55}")
print(f"  Degree sequence: {n_deg_coll:,} collision groups")
print(f"  + walk_weight_k4: {n_ww_coll:,} collision groups")
print(f"  + eigenvalues: {n_eig_coll} collision groups")
print(f"  + betweenness: {final_coll} collision groups")

if final_coll == 0:
    print(f"\n*** COMPLETE! degree + walk_weight_k4 + eigenvalue + betweenness")
    print(f"    distinguishes ALL {n_total:,} non-isomorphic 10-vertex graphs!")
    print(f"    Cost: O(N^4). Previous best: k=7 subgraph counting O(N^7).")
    print(f"    This is 1000x CHEAPER for N=10. ***")
else:
    print(f"\n  {final_coll} truly hard pairs remain at n=10.")
