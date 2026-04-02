"""
Test walk_weight_k4 on n=9 hard pairs.
n=9: k=5 fails with 250 collision groups. Does walk_weight_k4 resolve them?
If yes: k=5 + walk_weight_k4 = complete for n=9 too!
"""
import numpy as np
from collections import defaultdict
import time

def parse_graph6(line):
    line = line.strip()
    data = [ord(c)-63 for c in line]
    if data[0]<=62: n=data[0]; bs=1
    else: n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63); bs=4
    A = np.zeros((n,n), dtype=np.int8)
    bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6; bw=5-(bi%6)
            if bp<len(data) and (data[bp]>>bw)&1: A[i,j]=A[j,i]=1
            bi+=1
    return A

def walk_weight_k4(A):
    A4 = np.linalg.matrix_power(A.astype(int), 4)
    return tuple(sorted([tuple(sorted(A4[i])) for i in range(A.shape[0])]))

def degree_seq(A):
    return tuple(sorted(np.sum(A, axis=1)))

# Load n=9
g9_path = r"C:\Users\salih\Desktop\cr-universal\graph9.g6"
print("Loading n=9 graphs...")
t0 = time.time()
graphs = []
with open(g9_path) as f:
    for line in f:
        if line.strip():
            graphs.append(parse_graph6(line))
N = len(graphs)
print(f"Loaded {N} graphs in {time.time()-t0:.1f}s")

# Step 1: Group by degree sequence (fast, coarse)
print("\nStep 1: Degree sequence grouping...")
t0 = time.time()
deg_groups = defaultdict(list)
for i, A in enumerate(graphs):
    deg_groups[degree_seq(A)].append(i)
    if (i+1) % 50000 == 0: print(f"  {i+1}/{N}...", flush=True)
n_deg_groups = sum(1 for v in deg_groups.values() if len(v) > 1)
print(f"  {len(deg_groups)} unique degree seqs, {n_deg_groups} collision groups")
print(f"  Done in {time.time()-t0:.1f}s")

# Step 2: Within each degree collision group, compute walk_weight_k4
print("\nStep 2: walk_weight_k4 on degree-collision groups...")
t0 = time.time()
ww_collisions = 0
ww_resolved = 0
total_pairs = 0

for deg, indices in deg_groups.items():
    if len(indices) <= 1: continue

    # Compute walk_weight for this group
    ww_map = defaultdict(list)
    for i in indices:
        ww = walk_weight_k4(graphs[i])
        ww_map[ww].append(i)

    # Count remaining collisions
    for ww, ww_indices in ww_map.items():
        if len(ww_indices) > 1:
            ww_collisions += 1

    group_pairs = len(indices) * (len(indices)-1) // 2
    resolved_pairs = group_pairs - sum(len(v)*(len(v)-1)//2 for v in ww_map.values() if len(v)>1)
    ww_resolved += resolved_pairs
    total_pairs += group_pairs

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

print(f"\nRESULTS:")
print(f"  Degree sequence collision groups: {n_deg_groups}")
print(f"  After walk_weight_k4: {ww_collisions} remaining collisions")
print(f"  Resolved: {ww_resolved}/{total_pairs} pairs ({ww_resolved/max(total_pairs,1)*100:.1f}%)")

if ww_collisions == 0:
    print(f"\n*** degree_seq + walk_weight_k4 = COMPLETE for n=9! ***")
else:
    print(f"\n  {ww_collisions} groups still unresolved by walk_weight_k4.")
    print(f"  Need k=5 or another feature for these.")
