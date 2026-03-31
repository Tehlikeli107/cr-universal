"""
Lens Discovery v2: use WEAK invariants to create hard pairs,
then find which NEW features resolve them.

Weak = degree sequence only (loses eigenvalue info).
This creates MORE hard pairs = more opportunity for new lens.
"""
import torch
import numpy as np
from itertools import combinations
import time
import networkx as nx

DEVICE = torch.device('cuda')

def degree_seq(adj):
    return tuple(sorted(adj.sum(dim=1).int().tolist()))

def num_edges(adj):
    return int(adj.sum().item() // 2)

def num_triangles(adj):
    A = adj.float()
    return round((A @ A @ A).trace().item() / 6)

def weak_invariant(adj):
    """Only degree sequence + edge count + triangle count. Deliberately WEAK."""
    return (degree_seq(adj), num_edges(adj), num_triangles(adj))

# Candidate NEW features
def eigenvalue_sig(adj):
    try:
        eigs = torch.linalg.eigvalsh(adj.float())
        return tuple(sorted([round(e.item(), 3) for e in eigs]))
    except: return None

def local_clustering(adj):
    N = adj.shape[0]
    coeffs = []
    for v in range(N):
        nb = [u for u in range(N) if adj[v,u]]
        if len(nb) < 2: coeffs.append(0.0); continue
        edges = sum(1 for i,j in combinations(nb, 2) if adj[i,j])
        coeffs.append(round(edges / (len(nb)*(len(nb)-1)/2), 3))
    return tuple(sorted(coeffs))

def betweenness_sig(adj):
    N = adj.shape[0]
    G = nx.Graph()
    for i in range(N):
        for j in range(i+1, N):
            if adj[i,j]: G.add_edge(i, j)
    bc = nx.betweenness_centrality(G)
    return tuple(sorted([round(v, 3) for v in bc.values()]))

def resistance_sig(adj):
    N = adj.shape[0]
    L = torch.diag(adj.sum(dim=1).float()) - adj.float()
    try:
        Lp = torch.linalg.pinv(L)
        dists = []
        for i in range(N):
            for j in range(i+1,N):
                dists.append(round((Lp[i,i]+Lp[j,j]-2*Lp[i,j]).item(), 3))
        return tuple(sorted(dists))
    except: return None

def path4_count(adj):
    A = adj.float()
    A4 = A @ A @ A @ A
    return tuple(sorted([round(A4[v].sum().item()) for v in range(A.shape[0])]))

def neighborhood_hash(adj):
    """Two-round WL-like hash."""
    N = adj.shape[0]
    colors = adj.sum(dim=1).int().tolist()  # degree
    for _ in range(2):
        new_colors = []
        for v in range(N):
            nb_colors = sorted([colors[u] for u in range(N) if adj[v,u]])
            new_colors.append(hash((colors[v], tuple(nb_colors))) % 10**9)
        colors = new_colors
    return tuple(sorted(colors))

def edge_triangle_profile(adj):
    """For each edge: how many triangles it participates in."""
    N = adj.shape[0]
    profile = []
    for u in range(N):
        for v in range(u+1, N):
            if not adj[u,v]: continue
            tri = sum(1 for w in range(N) if w!=u and w!=v and adj[u,w] and adj[v,w])
            profile.append(tri)
    return tuple(sorted(profile))

CANDIDATES = [
    ("eigenvalues", eigenvalue_sig),
    ("local_clustering", local_clustering),
    ("betweenness", betweenness_sig),
    ("resistance_dist", resistance_sig),
    ("path4_count", path4_count),
    ("2-round WL hash", neighborhood_hash),
    ("edge_triangle", edge_triangle_profile),
]

# Load n=8
print("LENS DISCOVERY v2: Weak baseline, strong candidates")
print("=" * 55)

g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
graphs = []
with open(g6_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        G = nx.from_graph6_bytes(line.encode())
        N = G.number_of_nodes()
        adj = torch.zeros(N, N, device=DEVICE)
        for u, v in G.edges(): adj[u,v] = adj[v,u] = 1
        graphs.append(adj)

print(f"Loaded {len(graphs)} graphs on 8 vertices")

# Step 1: Weak invariants
print("\nStep 1: Computing WEAK invariants (degree + edges + triangles only)...")
t0 = time.time()
groups = {}
for i, adj in enumerate(graphs):
    inv = weak_invariant(adj)
    key = str(inv)
    if key not in groups: groups[key] = []
    groups[key].append(i)
print(f"  Done in {time.time()-t0:.1f}s")

hard = {k: v for k, v in groups.items() if len(v) > 1}
n_pairs = sum(len(v)*(len(v)-1)//2 for v in hard.values())
print(f"  Hard groups: {len(hard)} ({n_pairs} hard pairs)")
print(f"  Largest group: {max(len(v) for v in hard.values())} graphs")

# Step 2: Test candidates on sample of hard pairs
print(f"\nStep 2: Testing {len(CANDIDATES)} candidate lenses...")

# Sample hard pairs for speed
sample_pairs = []
for key, indices in hard.items():
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            sample_pairs.append((indices[i], indices[j]))
            if len(sample_pairs) >= 200: break
        if len(sample_pairs) >= 200: break
    if len(sample_pairs) >= 200: break

print(f"  Testing on {len(sample_pairs)} hard pairs (sampled)")

for feat_name, feat_fn in CANDIDATES:
    n_dist = 0
    n_fail = 0
    t0 = time.time()
    for gi, gj in sample_pairs:
        try:
            fi = feat_fn(graphs[gi])
            fj = feat_fn(graphs[gj])
            if fi is not None and fj is not None and fi != fj:
                n_dist += 1
        except:
            n_fail += 1
    elapsed = time.time() - t0
    pct = n_dist / len(sample_pairs) * 100 if sample_pairs else 0
    tag = "*** NEW LENS! ***" if pct > 50 else ("partial" if pct > 0 else "useless")
    print(f"  {feat_name:20s}: {n_dist:4d}/{len(sample_pairs)} ({pct:5.1f}%) {elapsed:5.1f}s  [{tag}]")

print(f"\n  Features that distinguish >50% of hard pairs = STRONG new lens")
print(f"  Features that distinguish >0% = carry SOME new info")
print(f"  Features that distinguish 0% = redundant with weak baseline")
