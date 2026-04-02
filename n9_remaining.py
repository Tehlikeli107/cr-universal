"""Analyze the 287 remaining collision groups at n=9."""
import numpy as np
from collections import defaultdict

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

graphs = []
with open(r"C:\Users\salih\Desktop\cr-universal\graph9.g6") as f:
    for line in f:
        if line.strip(): graphs.append(parse_graph6(line))

print(f"Finding remaining collisions in {len(graphs)} n=9 graphs...")

# Combined key: degree + walk_weight
groups = defaultdict(list)
for i, A in enumerate(graphs):
    key = (degree_seq(A), walk_weight_k4(A))
    groups[key].append(i)
    if (i+1) % 50000 == 0: print(f"  {i+1}/{len(graphs)}...", flush=True)

hard = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\n{len(hard)} collision groups remain.")

# Can eigenvalues resolve them?
eig_resolved = 0
eig_total = 0
for key, indices in hard.items():
    eig_map = defaultdict(list)
    for i in indices:
        eigs = tuple(sorted(np.round(np.linalg.eigvalsh(graphs[i].astype(float)), 4)))
        eig_map[eigs].append(i)
    for eigs, eidx in eig_map.items():
        if len(eidx) > 1: eig_total += 1
    if len(eig_map) == len(indices):
        eig_resolved += 1

print(f"Eigenvalues resolve {eig_resolved}/{len(hard)} groups ({eig_resolved/len(hard)*100:.1f}%)")
print(f"Remaining after eigenvalues: {len(hard)-eig_resolved}")

# Are any COSPECTRAL? (same eigenvalues AND same walk_weight = truly hard)
truly_hard = len(hard) - eig_resolved
if truly_hard > 0:
    print(f"\n*** {truly_hard} groups are COSPECTRAL + same walk_weight = TRULY HARD ***")
    print(f"These pairs need k=6 or higher to resolve.")
else:
    print(f"\n*** degree + walk_weight + eigenvalues = COMPLETE for n=9! ***")
    print(f"No truly hard pairs remain.")
