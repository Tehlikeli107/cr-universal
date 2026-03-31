"""Test: is walk_weight_k4 COMPLETE for n=8? (12346 graphs, 12346 unique?)"""
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
    profiles = tuple(sorted([tuple(sorted(A4[i])) for i in range(A.shape[0])]))
    return profiles

def vertex_del_tr4(A):
    n = A.shape[0]
    traces = []
    for v in range(n):
        idx = [i for i in range(n) if i != v]
        Av = A[np.ix_(idx, idx)]
        Av4 = np.linalg.matrix_power(Av.astype(int), 4)
        traces.append(int(np.trace(Av4)))
    return tuple(sorted(traces))

g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
graphs = []
with open(g6_path) as f:
    for line in f:
        if line.strip(): graphs.append(parse_graph6(line))

N = len(graphs)
print(f"Testing new lenses on ALL {N} n=8 graphs")
print("=" * 55)

for lens_name, lens_fn in [("walk_weight_k4", walk_weight_k4), ("vertex_del_tr4", vertex_del_tr4)]:
    print(f"\n{lens_name}:")
    t0 = time.time()
    groups = defaultdict(list)
    for i, A in enumerate(graphs):
        fp = lens_fn(A)
        groups[fp].append(i)
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{N}...", flush=True)
    elapsed = time.time() - t0

    n_unique = len(groups)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    n_coll_groups = len(collisions)
    n_coll_pairs = sum(len(v)*(len(v)-1)//2 for v in collisions.values())

    print(f"  Unique: {n_unique}/{N} ({n_unique/N*100:.1f}%)")
    print(f"  Collision groups: {n_coll_groups}")
    print(f"  Collision pairs: {n_coll_pairs}")
    print(f"  Time: {elapsed:.1f}s")

    if n_coll_groups == 0:
        print(f"  *** COMPLETE INVARIANT! All {N} graphs distinguished! ***")
    elif n_coll_groups <= 10:
        print(f"  Near-complete. {n_coll_groups} groups remain:")
        for fp, indices in list(collisions.items())[:5]:
            print(f"    Group: graphs {indices}")
