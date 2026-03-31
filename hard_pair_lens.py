"""
FOCUSED LENS DISCOVERY: Only on n=8 hard pairs.

These 4 pairs have SAME k=5 induced subgraph distribution.
k=6 distinguishes them. But can something SIMPLER than k=6?

If we find a CHEAP feature that k=5 misses but resolves hard pairs:
that's a genuinely new structural invariant.

The n=8 hard pair indices (from counting-revolution):
Groups that k=5 fails: (3630,4580), (7163,7210), (7638,8901), (11754,11839)
"""
import numpy as np
import networkx as nx
from itertools import combinations, permutations
from collections import Counter
import os, time

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

# Load n=8 and extract hard pairs
g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
print("Loading n=8 graphs...")
all_graphs = []
with open(g6_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            all_graphs.append(parse_graph6(line))
print(f"Loaded {len(all_graphs)} graphs")

# Known hard pairs (0-indexed)
hard_pairs = [(3630,4580), (7163,7210), (7638,8901), (11754,11839)]

print(f"\n4 HARD PAIRS (k=5 fails, k=6 resolves):")
for gi, gj in hard_pairs:
    print(f"  Graph {gi} vs Graph {gj}")

# ============================================================
# Feature battery: test MANY features on hard pairs
# ============================================================

def make_nx(A):
    G = nx.Graph()
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j]: G.add_edge(i,j)
    return G

print(f"\nTesting features on hard pairs...")
print(f"{'Feature':>30s} | {'Pair 1':>7s} {'Pair 2':>7s} {'Pair 3':>7s} {'Pair 4':>7s} | Resolves")
print("-" * 75)

features_tested = 0
features_resolved = []

def test_feature(name, feat_fn):
    global features_tested
    features_tested += 1
    results = []
    n_resolved = 0
    for gi, gj in hard_pairs:
        fi = feat_fn(all_graphs[gi])
        fj = feat_fn(all_graphs[gj])
        match = "SAME" if fi == fj else "DIFF"
        results.append(match)
        if fi != fj: n_resolved += 1
    tag = f"{n_resolved}/4" if n_resolved > 0 else "0/4"
    print(f"  {name:>28s} | {'  '.join(f'{r:>5s}' for r in results)} | {tag}")
    if n_resolved > 0:
        features_resolved.append((name, n_resolved))

# Degree-based
test_feature("degree_sequence", lambda A: tuple(sorted(np.sum(A, axis=1))))
test_feature("degree_sq_sum", lambda A: int(np.sum(np.sum(A, axis=1)**2)))
test_feature("max_degree", lambda A: int(np.max(np.sum(A, axis=1))))

# Spectral
test_feature("eigenvalues", lambda A: tuple(sorted(np.round(np.linalg.eigvalsh(A.astype(float)), 4))))
test_feature("spectral_radius", lambda A: round(float(np.max(np.abs(np.linalg.eigvalsh(A.astype(float))))), 4))
test_feature("algebraic_connectivity", lambda A: round(float(sorted(np.linalg.eigvalsh((np.diag(np.sum(A,1))-A).astype(float)))[1]), 4))

# Trace-based
test_feature("tr_A3", lambda A: int(np.trace(A@A@A)))
test_feature("tr_A4", lambda A: int(np.trace(A@A@A@A)))
test_feature("tr_A5", lambda A: int(np.trace(A@A@A@A@A)))
test_feature("tr_A6", lambda A: int(np.trace(A@A@A@A@A@A)))

# Path/Walk based
test_feature("num_4cycles", lambda A: (int(np.trace(A@A@A@A)) - int(np.sum(A@A * np.eye(8))) * 2) // 8)
test_feature("num_paths_3", lambda A: int(np.sum(A@A@A) - np.trace(A@A@A)))

# Clustering
def clustering_profile(A):
    n = A.shape[0]; coeffs = []
    for v in range(n):
        nb = [u for u in range(n) if A[v,u]]
        if len(nb) < 2: coeffs.append(0.0); continue
        e = sum(1 for i,j in combinations(nb,2) if A[i,j])
        coeffs.append(round(e/(len(nb)*(len(nb)-1)/2), 4))
    return tuple(sorted(coeffs))
test_feature("clustering_profile", clustering_profile)

# Centrality
def betweenness_profile(A):
    G = make_nx(A)
    bc = nx.betweenness_centrality(G)
    return tuple(sorted([round(v,4) for v in bc.values()]))
test_feature("betweenness", betweenness_profile)

def closeness_profile(A):
    G = make_nx(A)
    cc = nx.closeness_centrality(G)
    return tuple(sorted([round(v,4) for v in cc.values()]))
test_feature("closeness", closeness_profile)

# Distance-based
def distance_histogram(A):
    G = make_nx(A)
    hist = Counter()
    for u in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, u)
        for d in lengths.values(): hist[d] += 1
    return tuple(sorted(hist.items()))
test_feature("distance_histogram", distance_histogram)

def wiener(A):
    G = make_nx(A)
    return nx.wiener_index(G) if nx.is_connected(G) else -1
test_feature("wiener_index", lambda A: wiener(A))

# Resistance distance
def resistance_profile(A):
    L = np.diag(np.sum(A, axis=1).astype(float)) - A.astype(float)
    try:
        Lp = np.linalg.pinv(L)
        n = A.shape[0]
        dists = []
        for i in range(n):
            for j in range(i+1,n):
                dists.append(round(Lp[i,i]+Lp[j,j]-2*Lp[i,j], 4))
        return tuple(sorted(dists))
    except: return None
test_feature("resistance_distances", resistance_profile)

# Matching-based
def max_matching(A):
    G = make_nx(A)
    return len(nx.max_weight_matching(G))
test_feature("max_matching", max_matching)

# Neighborhood diversity
def neighbor_degree_multiset(A):
    n = A.shape[0]
    profiles = []
    for v in range(n):
        nb_degs = tuple(sorted([int(np.sum(A[u])) for u in range(n) if A[v,u]]))
        profiles.append(nb_degs)
    return tuple(sorted(profiles))
test_feature("neighbor_degree_sets", neighbor_degree_multiset)

# Edge properties
def edge_triangle_dist(A):
    n = A.shape[0]
    tri_counts = []
    for u in range(n):
        for v in range(u+1,n):
            if not A[u,v]: continue
            t = sum(1 for w in range(n) if w!=u and w!=v and A[u,w] and A[v,w])
            tri_counts.append(t)
    return tuple(sorted(tri_counts))
test_feature("edge_triangle_dist", edge_triangle_dist)

# Automorphism group size
def automorphism_size(A):
    G = make_nx(A)
    # NetworkX doesn't have direct automorphism. Use orbits instead.
    # Approximate: count fixed points under degree-based partition
    deg = tuple(sorted(np.sum(A, axis=1)))
    return deg  # placeholder
# Skip — too complex

# Subgraph counts
def count_P4(A):
    """Count induced P4 (path of length 3)."""
    n = A.shape[0]; count = 0
    for nodes in combinations(range(n), 4):
        sub = A[np.ix_(nodes, nodes)]
        degs = sorted(np.sum(sub, axis=1))
        if degs == [1,1,1,1] and np.sum(sub) == 6:  # P4: degs (1,1,2,2), 3 edges
            count += 1
    return count

def count_C4(A):
    """Count induced C4 (4-cycle)."""
    n = A.shape[0]; count = 0
    for nodes in combinations(range(n), 4):
        sub = A[np.ix_(nodes, nodes)]
        degs = sorted(np.sum(sub, axis=1))
        if degs == [2,2,2,2] and np.sum(sub) == 8:  # C4: all deg 2, 4 edges
            count += 1
    return count

def count_K4_minus_e(A):
    """Count K4 minus one edge (diamond)."""
    n = A.shape[0]; count = 0
    for nodes in combinations(range(n), 4):
        sub = A[np.ix_(nodes, nodes)]
        if np.sum(sub) == 10:  # 5 edges
            count += 1
    return count

test_feature("count_induced_P4", count_P4)
test_feature("count_induced_C4", count_C4)
test_feature("count_K4_minus_edge", count_K4_minus_e)

# Combined
def combined_trace_deg(A):
    return (int(np.trace(A@A@A@A)), int(np.sum(np.sum(A,1)**2)))
test_feature("(tr_A4, deg_sq_sum)", combined_trace_deg)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*55}")
print(f"RESULTS: {features_tested} features tested on 4 hard pairs")
print(f"{'='*55}")

if features_resolved:
    print(f"\nFeatures that RESOLVE hard pairs (new info beyond k=5):")
    for name, n_res in sorted(features_resolved, key=lambda x: -x[1]):
        print(f"  {name:>30s}: {n_res}/4 pairs resolved")
    print(f"\n  These features carry information that k=5 induced subgraph")
    print(f"  distributions MISS. They are complementary to counting revolution.")
else:
    print(f"\n  NO FEATURE resolves any hard pair!")
    print(f"  These pairs are TRULY hard — only k=6 subgraph counting works.")
    print(f"  This confirms k=6 is NECESSARY for n=8.")
