"""Analyze 9 truly hard groups: cospectral + same walk_weight at n=9.
Can ANY cheap feature resolve them? Or do they need k=6?"""
import numpy as np
from collections import defaultdict
from itertools import combinations
import networkx as nx

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

def ww4(A): A4=np.linalg.matrix_power(A.astype(int),4); return tuple(sorted([tuple(sorted(A4[i])) for i in range(A.shape[0])]))
def deg(A): return tuple(sorted(np.sum(A,1)))
def eig(A): return tuple(sorted(np.round(np.linalg.eigvalsh(A.astype(float)),4)))

graphs=[]
with open(r"C:\Users\salih\Desktop\cr-universal\graph9.g6") as f:
    for line in f:
        if line.strip(): graphs.append(parse_graph6(line))

print(f"Finding 9 hard groups from {len(graphs)} n=9 graphs...")
groups=defaultdict(list)
for i,A in enumerate(graphs):
    key=(deg(A),ww4(A),eig(A)); groups[key].append(i)
    if (i+1)%50000==0: print(f"  {i+1}...",flush=True)

hard={k:v for k,v in groups.items() if len(v)>1}
print(f"\n{len(hard)} truly hard groups found.\n")

# Analyze each hard group
for gi, (key, indices) in enumerate(hard.items()):
    print(f"=== Hard Group {gi+1}: graphs {indices} ===")
    deg_seq = key[0]
    n_edges = sum(deg_seq)//2
    print(f"  Degree seq: {deg_seq}, edges: {n_edges}")

    # Test additional features
    for feat_name, feat_fn in [
        ("tr(A^5)", lambda A: int(np.trace(np.linalg.matrix_power(A.astype(int),5)))),
        ("tr(A^6)", lambda A: int(np.trace(np.linalg.matrix_power(A.astype(int),6)))),
        ("walk_weight_k5", lambda A: tuple(sorted([tuple(sorted(np.linalg.matrix_power(A.astype(int),5)[i])) for i in range(9)]))),
        ("walk_weight_k6", lambda A: tuple(sorted([tuple(sorted(np.linalg.matrix_power(A.astype(int),6)[i])) for i in range(9)]))),
        ("diameter", lambda A: nx.diameter(nx.Graph(A)) if nx.is_connected(nx.Graph(A)) else -1),
        ("clustering", lambda A: tuple(sorted([round(v,4) for v in nx.clustering(nx.Graph(A)).values()]))),
        ("betweenness", lambda A: tuple(sorted([round(v,4) for v in nx.betweenness_centrality(nx.Graph(A)).values()]))),
        ("vertex_del_tr4", lambda A: tuple(sorted([int(np.trace(np.linalg.matrix_power(np.delete(np.delete(A,v,0),v,1).astype(int),4))) for v in range(9)]))),
    ]:
        vals = [feat_fn(graphs[i]) for i in indices]
        unique = len(set([str(v) for v in vals]))
        status = "RESOLVES!" if unique == len(indices) else f"SAME ({unique}/{len(indices)})"
        if unique > 1 and unique < len(indices): status = f"PARTIAL ({unique}/{len(indices)})"
        print(f"  {feat_name:20s}: {status}")

    print()

# Summary
print("SUMMARY: Which features resolve the 9 truly hard groups?")
resolvers = defaultdict(int)
for key, indices in hard.items():
    for feat_name, feat_fn in [
        ("walk_weight_k5", lambda A: tuple(sorted([tuple(sorted(np.linalg.matrix_power(A.astype(int),5)[i])) for i in range(9)]))),
        ("walk_weight_k6", lambda A: tuple(sorted([tuple(sorted(np.linalg.matrix_power(A.astype(int),6)[i])) for i in range(9)]))),
        ("vertex_del_tr4", lambda A: tuple(sorted([int(np.trace(np.linalg.matrix_power(np.delete(np.delete(A,v,0),v,1).astype(int),4))) for v in range(9)]))),
        ("betweenness", lambda A: tuple(sorted([round(v,4) for v in nx.betweenness_centrality(nx.Graph(A)).values()]))),
    ]:
        vals = [feat_fn(graphs[i]) for i in indices]
        if len(set([str(v) for v in vals])) == len(indices):
            resolvers[feat_name] += 1

print(f"\n  {'Feature':>20s} | Resolves X/9 groups")
for name, count in sorted(resolvers.items(), key=lambda x:-x[1]):
    print(f"  {name:>20s} | {count}/9")
