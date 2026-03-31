"""
NEW LENS: Find a structural feature that existing lenses miss.

Strategy: Look at what makes hard pairs ACTUALLY different.
Not test random features — UNDERSTAND the difference first.
Then build a lens that captures THAT specific difference.

n=8 hard pairs: (3630,4580), (7163,7210), (7638,8901), (11754,11839)
All have SAME: degree seq, triangle count, k=5 subgraph distribution
All have DIFF: eigenvalues, k=6 subgraph distribution

Question: WHAT structural difference do eigenvalues capture
that degree+triangles+k=5 miss?

Answer: eigenvalues capture SPECTRAL SHAPE — how the graph
"vibrates" at different frequencies. Two graphs can have same
local structure (same degrees, same triangles) but different
GLOBAL shape (different spectral resonances).

So the new lens needs to capture GLOBAL shape cheaply.
Eigenvalues cost O(N^3). k=6 costs O(N^6). Can we do O(N^2)?
"""
import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter

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

# Load hard pairs
g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
all_g = []
with open(g6_path) as f:
    for line in f:
        if line.strip(): all_g.append(parse_graph6(line))

pairs = [(3630,4580), (7163,7210), (7638,8901), (11754,11839)]

print("UNDERSTANDING HARD PAIRS")
print("=" * 55)

for pi, (gi, gj) in enumerate(pairs):
    A1, A2 = all_g[gi], all_g[gj]
    print(f"\nPair {pi+1}: graph {gi} vs {gj}")

    # What's SAME
    d1 = tuple(sorted(np.sum(A1,1)))
    d2 = tuple(sorted(np.sum(A2,1)))
    print(f"  Degree seq: {'SAME' if d1==d2 else 'DIFF'} {d1}")

    t1 = int(np.trace(A1@A1@A1)//6)
    t2 = int(np.trace(A2@A2@A2)//6)
    print(f"  Triangles: {'SAME' if t1==t2 else 'DIFF'} ({t1})")

    # What's DIFFERENT — eigenvalues
    e1 = np.round(sorted(np.linalg.eigvalsh(A1.astype(float))), 4)
    e2 = np.round(sorted(np.linalg.eigvalsh(A2.astype(float))), 4)
    print(f"  Eigenvalues: {'SAME' if np.allclose(e1,e2) else 'DIFF'}")

    # VISUALIZE the actual difference
    G1 = nx.Graph(A1); G2 = nx.Graph(A2)

    # Diameter
    diam1 = nx.diameter(G1) if nx.is_connected(G1) else -1
    diam2 = nx.diameter(G2) if nx.is_connected(G2) else -1
    print(f"  Diameter: {diam1} vs {diam2} {'SAME' if diam1==diam2 else 'DIFF'}")

    # Girth (shortest cycle)
    girth1 = nx.girth(G1) if nx.is_connected(G1) else -1
    girth2 = nx.girth(G2) if nx.is_connected(G2) else -1
    print(f"  Girth: {girth1} vs {girth2} {'SAME' if girth1==girth2 else 'DIFF'}")

    # Number of spanning trees (Kirchhoff)
    def n_spanning_trees(A):
        n = A.shape[0]
        L = np.diag(np.sum(A, axis=1).astype(float)) - A.astype(float)
        return round(abs(np.linalg.det(L[1:,1:])))
    st1 = n_spanning_trees(A1); st2 = n_spanning_trees(A2)
    print(f"  Spanning trees: {st1} vs {st2} {'SAME' if st1==st2 else 'DIFF'}")

    # Distance distribution (BFS from each vertex)
    def distance_multiset(A):
        n = A.shape[0]; G = nx.Graph(A)
        all_d = []
        for v in range(n):
            lengths = dict(nx.single_source_shortest_path_length(G, v))
            all_d.append(tuple(sorted(lengths.values())))
        return tuple(sorted(all_d))
    dm1 = distance_multiset(A1); dm2 = distance_multiset(A2)
    print(f"  Distance multisets: {'SAME' if dm1==dm2 else 'DIFF'}")

    # ===== NEW IDEA: "walk diversity" at each vertex =====
    # For each vertex v: compute the SET of distinct walks of length k
    # Not just count — the DIVERSITY of where you can reach
    def walk_diversity(A, k=4):
        """For each vertex: how many DISTINCT vertices reachable in exactly k steps?"""
        n = A.shape[0]
        Ak = np.linalg.matrix_power(A.astype(int), k)
        # Ak[i,j] = number of walks from i to j of length k
        # diversity = number of j where Ak[i,j] > 0
        diversity = [(Ak[i] > 0).sum() for i in range(n)]
        return tuple(sorted(diversity))

    wd1_3 = walk_diversity(A1, 3); wd2_3 = walk_diversity(A2, 3)
    wd1_4 = walk_diversity(A1, 4); wd2_4 = walk_diversity(A2, 4)
    wd1_5 = walk_diversity(A1, 5); wd2_5 = walk_diversity(A2, 5)
    print(f"  Walk diversity k=3: {'SAME' if wd1_3==wd2_3 else 'DIFF'}")
    print(f"  Walk diversity k=4: {'SAME' if wd1_4==wd2_4 else 'DIFF'}")
    print(f"  Walk diversity k=5: {'SAME' if wd1_5==wd2_5 else 'DIFF'}")

    # ===== NEW IDEA: "walk weight distribution" =====
    # Not just CAN you reach j in k steps, but HOW MANY WAYS
    def walk_weight_dist(A, k=4):
        """Distribution of walk counts: sorted Ak[i,:] for each i."""
        Ak = np.linalg.matrix_power(A.astype(int), k)
        profiles = [tuple(sorted(Ak[i])) for i in range(A.shape[0])]
        return tuple(sorted(profiles))

    ww1_4 = walk_weight_dist(A1, 4); ww2_4 = walk_weight_dist(A2, 4)
    ww1_5 = walk_weight_dist(A1, 5); ww2_5 = walk_weight_dist(A2, 5)
    print(f"  Walk weight k=4: {'SAME' if ww1_4==ww2_4 else 'DIFF'}")
    print(f"  Walk weight k=5: {'SAME' if ww1_5==ww2_5 else 'DIFF'}")

    # ===== NEW IDEA: "heat kernel signature" =====
    # Heat kernel: e^{-tL} where L = laplacian
    # At each vertex: trace of heat kernel restricted to neighbors
    def heat_signature(A, t=1.0):
        n = A.shape[0]
        L = np.diag(np.sum(A, axis=1).astype(float)) - A.astype(float)
        H = np.linalg.matrix_power(np.eye(n) - t/10 * L, 10)  # approx e^{-tL}
        # Per-vertex: diagonal of H
        sig = tuple(sorted([round(H[i,i], 4) for i in range(n)]))
        return sig

    hs1 = heat_signature(A1, 1.0); hs2 = heat_signature(A2, 1.0)
    print(f"  Heat kernel t=1: {'SAME' if hs1==hs2 else 'DIFF'}")
    hs1_2 = heat_signature(A1, 2.0); hs2_2 = heat_signature(A2, 2.0)
    print(f"  Heat kernel t=2: {'SAME' if hs1_2==hs2_2 else 'DIFF'}")

    # ===== NEW IDEA: "neighborhood overlap matrix" =====
    # For each pair of vertices: how many common neighbors?
    # Distribution of these counts
    def common_neighbor_dist(A):
        n = A.shape[0]
        cn = A.astype(int) @ A.astype(int)  # cn[i,j] = # common neighbors
        # Extract upper triangle
        vals = [cn[i,j] for i in range(n) for j in range(i+1,n)]
        return tuple(sorted(vals))
    cn1 = common_neighbor_dist(A1); cn2 = common_neighbor_dist(A2)
    print(f"  Common neighbor dist: {'SAME' if cn1==cn2 else 'DIFF'}")

    # ===== NEW IDEA: "vertex deleted subgraph trace" =====
    # For each vertex v: compute tr(A_v^3) where A_v = adjacency without v
    def vertex_deleted_traces(A, k=3):
        n = A.shape[0]
        traces = []
        for v in range(n):
            idx = [i for i in range(n) if i != v]
            Av = A[np.ix_(idx, idx)]
            Avk = np.linalg.matrix_power(Av.astype(int), k)
            traces.append(int(np.trace(Avk)))
        return tuple(sorted(traces))

    vt1_3 = vertex_deleted_traces(A1, 3); vt2_3 = vertex_deleted_traces(A2, 3)
    vt1_4 = vertex_deleted_traces(A1, 4); vt2_4 = vertex_deleted_traces(A2, 4)
    vt1_5 = vertex_deleted_traces(A1, 5); vt2_5 = vertex_deleted_traces(A2, 5)
    print(f"  Vertex-deleted tr(A^3): {'SAME' if vt1_3==vt2_3 else 'DIFF'}")
    print(f"  Vertex-deleted tr(A^4): {'SAME' if vt1_4==vt2_4 else 'DIFF'}")
    print(f"  Vertex-deleted tr(A^5): {'SAME' if vt1_5==vt2_5 else 'DIFF'}")

# ===== SUMMARY =====
print(f"\n{'='*55}")
print("SUMMARY: Which features distinguish ALL 4 hard pairs?")
print("=" * 55)

features_all = []
features_some = []

for feat_name, feat_fn in [
    ("walk_diversity_k3", lambda A: walk_diversity(A,3)),
    ("walk_diversity_k4", lambda A: walk_diversity(A,4)),
    ("walk_diversity_k5", lambda A: walk_diversity(A,5)),
    ("walk_weight_k4", lambda A: walk_weight_dist(A,4)),
    ("walk_weight_k5", lambda A: walk_weight_dist(A,5)),
    ("heat_kernel_t1", lambda A: heat_signature(A,1.0)),
    ("heat_kernel_t2", lambda A: heat_signature(A,2.0)),
    ("common_neighbor_dist", common_neighbor_dist),
    ("vertex_del_tr3", lambda A: vertex_deleted_traces(A,3)),
    ("vertex_del_tr4", lambda A: vertex_deleted_traces(A,4)),
    ("vertex_del_tr5", lambda A: vertex_deleted_traces(A,5)),
    ("spanning_trees", lambda A: n_spanning_trees(A)),
    ("distance_multisets", distance_multiset),
]:
    n_resolved = 0
    for gi, gj in pairs:
        if feat_fn(all_g[gi]) != feat_fn(all_g[gj]):
            n_resolved += 1
    if n_resolved == 4:
        features_all.append(feat_name)
        print(f"  [4/4] {feat_name} -- COMPLETE (resolves ALL hard pairs)")
    elif n_resolved > 0:
        features_some.append((feat_name, n_resolved))
        print(f"  [{n_resolved}/4] {feat_name} -- partial")
    else:
        print(f"  [0/4] {feat_name}")

if features_all:
    print(f"\n*** NEW LENSES (4/4): {features_all} ***")
    print("These are CHEAPER than eigenvalues (O(N^3)) or k=6 (O(N^6))")
