"""Test: k=5 + walk_weight_k4 combined = complete?"""
import numpy as np
from collections import defaultdict
from itertools import combinations, permutations
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

def build_k5_lookup():
    k=5; edges=[(i,j) for i in range(k) for j in range(i+1,k)]
    ne=len(edges); canon={}; tid=0; arr=np.zeros(1<<ne, dtype=np.int32)
    for bits in range(1<<ne):
        A=np.zeros((k,k),dtype=np.int8)
        for idx,(i,j) in enumerate(edges):
            if (bits>>idx)&1: A[i,j]=A[j,i]=1
        mb=min(sum(int(A[p[i],p[j]])<<ei for ei,(i,j) in enumerate(edges)) for p in permutations(range(k)))
        if mb not in canon: canon[mb]=tid; tid+=1
        arr[bits]=canon[mb]
    return arr, tid

def k5_signature(A, lookup, n_types):
    n=A.shape[0]; k=5; edges=[(i,j) for i in range(k) for j in range(i+1,k)]
    counts=np.zeros(n_types, dtype=np.int32)
    for combo in combinations(range(n), k):
        bits=sum(int(A[combo[i],combo[j]])<<ei for ei,(i,j) in enumerate(edges))
        counts[lookup[bits]]+=1
    return tuple(counts)

def walk_weight_k4(A):
    A4=np.linalg.matrix_power(A.astype(int),4)
    return tuple(sorted([tuple(sorted(A4[i])) for i in range(A.shape[0])]))

g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
graphs=[]
with open(g6_path) as f:
    for line in f:
        if line.strip(): graphs.append(parse_graph6(line))
N=len(graphs)

print(f"COMBINED LENS TEST: k=5 + walk_weight_k4 on {N} n=8 graphs")
print("="*55)

# Build k=5 lookup
print("Building k=5 lookup...")
lookup, n_types = build_k5_lookup()
print(f"  {n_types} types")

# Compute combined signature
print("Computing combined signatures...")
t0=time.time()
groups=defaultdict(list)
for i,A in enumerate(graphs):
    sig_k5 = k5_signature(A, lookup, n_types)
    sig_ww = walk_weight_k4(A)
    combined = (sig_k5, sig_ww)
    groups[combined].append(i)
    if (i+1)%2000==0: print(f"  {i+1}/{N}...",flush=True)
elapsed=time.time()-t0

n_unique=len(groups)
collisions={k:v for k,v in groups.items() if len(v)>1}
n_coll=len(collisions)

print(f"\nRESULT: {n_unique}/{N} unique ({n_unique/N*100:.2f}%)")
print(f"Collisions: {n_coll}")
print(f"Time: {elapsed:.1f}s")

if n_coll==0:
    print(f"\n*** COMPLETE! k=5 + walk_weight_k4 = COMPLETE INVARIANT for n=8! ***")
    print(f"  This is CHEAPER than k=6: O(N^5 + N^4) vs O(N^6)")
    print(f"  NEW LENS: walk_weight_k4 COMPLEMENTS k=5 perfectly!")
else:
    print(f"\n  {n_coll} collisions remain. Not complete.")
    for fp, indices in list(collisions.items())[:5]:
        print(f"    Group: graphs {indices}")
