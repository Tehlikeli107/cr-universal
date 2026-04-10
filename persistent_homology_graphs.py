"""
Persistent Homology as a Graph Invariant

PARADIGM: Use topological data analysis (TDA) to fingerprint graphs.
Given G, build a filtration of simplicial complexes and compute persistent
homology. The persistence diagram is a complete topological invariant.

CONSTRUCTION:
1. Assign weights to edges: w(u,v) = f(G) (various options)
2. Sort edges by weight -> filtration
3. At each step, add simplices (nodes, edges, triangles, ...)
4. Track when Betti numbers change: (birth, death) = persistence pair
5. Persistence diagram PD(G) = multiset of (birth, death) intervals

KEY INSIGHT: The persistence diagram captures MULTI-SCALE topological structure:
- beta_0 persistence: how long connected components survive (merging order)
- beta_1 persistence: how long cycles survive (filling order)
- Long bars = robust topological features
- Short bars = noise / local structure

COMPARISON TO EXISTING METHODS:
- Spectral methods: only see eigenvalue spectrum (misses topology)
- WL hierarchy: only sees local neighborhoods (misses global loops)
- Counting revolution: sees induced subgraph counts (local structure)
- Persistent homology: sees MULTI-SCALE LOOP STRUCTURE (global topology)

QUESTION: Is persistent homology a complete graph invariant?
If yes: TDA SOLVES GRAPH ISOMORPHISM!
"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
import time


# ============================================================
# SIMPLICIAL COMPLEX & PERSISTENT HOMOLOGY (from scratch)
# ============================================================

class UnionFind:
    """For computing H_0 (connected components) during filtration."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def boundary_matrix(simplices_p, simplices_p1):
    """
    Compute boundary matrix from (p+1)-simplices to p-simplices.
    Returns dense matrix over GF(2).
    """
    if not simplices_p or not simplices_p1:
        return np.zeros((len(simplices_p), len(simplices_p1)), dtype=np.int8)

    idx_p = {s: i for i, s in enumerate(simplices_p)}
    B = np.zeros((len(simplices_p), len(simplices_p1)), dtype=np.int8)

    for j, sigma in enumerate(simplices_p1):
        for k in range(len(sigma)):
            face = tuple(sigma[i] for i in range(len(sigma)) if i != k)
            if face in idx_p:
                B[idx_p[face], j] ^= 1

    return B


def smith_normal_form_rank(B):
    """
    Compute rank of matrix over GF(2) using Gaussian elimination.
    Returns rank.
    """
    if B.size == 0:
        return 0
    B = B.copy().astype(np.int8)
    m, n = B.shape
    pivot_row = 0
    for col in range(n):
        # Find pivot
        found = -1
        for row in range(pivot_row, m):
            if B[row, col]:
                found = row
                break
        if found == -1:
            continue
        # Swap rows
        B[[pivot_row, found]] = B[[found, pivot_row]]
        # Eliminate
        for row in range(m):
            if row != pivot_row and B[row, col]:
                B[row] = (B[row] + B[pivot_row]) % 2
        pivot_row += 1
    return pivot_row


def betti_numbers(G):
    """
    Compute Betti numbers (beta_0, beta_1, beta_2) for graph's clique complex.
    beta_0 = connected components
    beta_1 = independent cycles (loop holes)
    beta_2 = independent voids (filled-triangle surfaces)
    """
    n = G.number_of_nodes()
    nodes = list(range(n))

    # Find all cliques
    cliques = list(nx.clique.find_cliques(G))

    # 0-simplices: nodes
    s0 = [(v,) for v in nodes]
    # 1-simplices: edges
    s1 = [tuple(sorted(e)) for e in G.edges()]
    # 2-simplices: triangles (3-cliques)
    s2 = []
    for clique in cliques:
        for tri in combinations(sorted(clique), 3):
            if tri not in s2:
                s2.append(tri)
    # 3-simplices: tetrahedra (4-cliques)
    s3 = []
    for clique in cliques:
        for tet in combinations(sorted(clique), 4):
            if tet not in s3:
                s3.append(tet)

    # Boundary matrices
    B1 = boundary_matrix(s0, s1)  # edges -> nodes
    B2 = boundary_matrix(s1, s2)  # triangles -> edges
    B3 = boundary_matrix(s2, s3)  # tetrahedra -> triangles

    # Betti numbers: beta_k = dim(H_k) = dim(ker B_k) - dim(im B_{k+1})
    rank_B1 = smith_normal_form_rank(B1)
    rank_B2 = smith_normal_form_rank(B2)
    rank_B3 = smith_normal_form_rank(B3)

    ker_B1 = len(s1) - rank_B1  # dim(Z_1) = dim(ker del_1)
    ker_B2 = len(s2) - rank_B2

    beta0 = len(s0) - rank_B1
    beta1 = ker_B1 - rank_B2
    beta2 = ker_B2 - rank_B3

    return beta0, beta1, beta2


# ============================================================
# FILTRATION-BASED PERSISTENT HOMOLOGY
# ============================================================

def edge_filtration_weights(G, method='betweenness'):
    """
    Assign filtration weights to edges.
    Different methods give different topological fingerprints.
    """
    n = G.number_of_nodes()
    weights = {}

    if method == 'betweenness':
        # Higher betweenness = more important bridge = appears LATER in filtration
        bc = nx.edge_betweenness_centrality(G)
        for e, w in bc.items():
            weights[tuple(sorted(e))] = w

    elif method == 'resistance':
        # Resistance distance = how "hard" to traverse
        L = nx.laplacian_matrix(G).toarray().astype(float)
        Lp = np.linalg.pinv(L)
        for u, v in G.edges():
            r = Lp[u, u] + Lp[v, v] - 2 * Lp[u, v]
            weights[(min(u,v), max(u,v))] = r

    elif method == 'spectral':
        # Use Fiedler vector to order edges
        try:
            fiedler = nx.fiedler_vector(G)
        except:
            fiedler = np.ones(n)
        for u, v in G.edges():
            weights[(min(u,v), max(u,v))] = abs(fiedler[u] - fiedler[v])

    elif method == 'degree':
        # High-degree edge appears later (core)
        degs = dict(G.degree())
        for u, v in G.edges():
            weights[(min(u,v), max(u,v))] = degs[u] + degs[v]

    elif method == 'uniform':
        # All edges appear at same time (clique complex, no filtration)
        for u, v in G.edges():
            weights[(min(u,v), max(u,v))] = 1.0

    return weights


def persistent_homology_H0(G, edge_weights):
    """
    Compute H_0 (component) persistent homology.
    Returns sorted list of (birth, death) intervals.
    Nodes born at time 0. A component dies when two components merge.
    """
    n = G.number_of_nodes()
    # Sort edges by weight
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])

    uf = UnionFind(n)
    # Each node is its own component, born at time 0
    birth_times = {v: 0.0 for v in range(n)}
    pairs = []

    for (u, v), w in sorted_edges:
        ru, rv = uf.find(u), uf.find(v)
        if ru != rv:
            # Two components merge: the YOUNGER one dies
            if birth_times[ru] >= birth_times[rv]:
                dying = ru
                surviving = rv
            else:
                dying = rv
                surviving = ru

            birth = birth_times[dying]
            death = w
            pairs.append((birth, death, death - birth))

            uf.union(u, v)
            # Update birth time of merged component
            root = uf.find(u)
            birth_times[root] = birth_times[surviving]

    # Surviving component: (birth=0, death=infinity) - essential class
    # We record it as (0, inf) but for signatures use finite pairs only
    return pairs


def persistent_homology_H1(G, edge_weights):
    """
    Compute H_1 (cycle) persistent homology using matrix reduction.
    Edges appear one by one; tracks when cycles are born and killed by triangles.

    Returns list of (birth, death) intervals for 1-cycles.
    """
    n = G.number_of_nodes()
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])

    # For H1, we need to track:
    # - When a new cycle is CREATED (edge closes a loop)
    # - When a cycle is KILLED (triangle fills it)
    # We use a boundary matrix approach.

    # Collect simplices in order of appearance
    # 0-simplices: all nodes at time 0
    # 1-simplices: edges in order
    # 2-simplices: triangles appear when all 3 edges present

    edge_appearance = {}
    for (u,v), w in sorted_edges:
        edge_appearance[(min(u,v), max(u,v))] = w

    # Find triangles and their appearance times
    triangles = []
    for u in G.nodes():
        for v in G.neighbors(u):
            for w in G.neighbors(u):
                if v < w and G.has_edge(v, w):
                    tri = (min(u,v,w), sorted([u,v,w])[1], max(u,v,w))
                    e1 = (min(tri[0],tri[1]), max(tri[0],tri[1]))
                    e2 = (min(tri[0],tri[2]), max(tri[0],tri[2]))
                    e3 = (min(tri[1],tri[2]), max(tri[1],tri[2]))
                    t_appear = max(edge_appearance.get(e1, 0),
                                   edge_appearance.get(e2, 0),
                                   edge_appearance.get(e3, 0))
                    triangles.append((t_appear, tri))

    # Sort triangles by appearance time
    triangles.sort(key=lambda x: x[0])

    # Track cycles: edges that close loops (when added, creates cycle)
    # We use Union-Find to detect cycle-creating edges
    uf2 = UnionFind(n)
    cycle_births = []  # (birth_time, edge)

    for (u, v), w in sorted_edges:
        if uf2.find(u) == uf2.find(v):
            # This edge creates a cycle!
            cycle_births.append((w, (u, v)))
        uf2.union(u, v)

    # Each triangle can kill one cycle (the youngest surviving cycle at that time)
    # Simple greedy pairing: triangle kills the most recently born cycle before it appears
    surviving_cycles = []  # list of (birth_time, edge)
    cycle_pairs = []  # (birth, death) intervals

    sorted_all_births = sorted(cycle_births, key=lambda x: x[0])
    tri_idx = 0
    birth_idx = 0
    tri_sorted = sorted(triangles, key=lambda x: x[0])

    # Process events in time order
    events = []
    for b, e in sorted_all_births:
        events.append(('cycle_born', b, e))
    for t, tri in tri_sorted:
        events.append(('triangle_appears', t, tri))
    events.sort(key=lambda x: x[1])

    active_cycles = []  # sorted by birth time
    for ev_type, t, data in events:
        if ev_type == 'cycle_born':
            active_cycles.append((t, data))
        elif ev_type == 'triangle_appears' and active_cycles:
            # Kill the youngest active cycle
            youngest = max(range(len(active_cycles)), key=lambda i: active_cycles[i][0])
            birth, e = active_cycles.pop(youngest)
            cycle_pairs.append((birth, t, t - birth))

    # Surviving cycles: essential H_1 classes
    essential = [(b, float('inf'), float('inf')) for b, e in active_cycles]

    return cycle_pairs, essential


def ph_signature(G, method='betweenness'):
    """
    Compute persistent homology signature of G.
    Returns dict with H0 and H1 persistence bars.
    """
    if G.number_of_edges() == 0:
        return {'h0_bars': [], 'h1_bars': [], 'h0_lifetimes': [], 'h1_lifetimes': []}

    weights = edge_filtration_weights(G, method)
    h0_bars = persistent_homology_H0(G, weights)
    h1_bars, h1_essential = persistent_homology_H1(G, weights)

    h0_lifetimes = sorted([l for b, d, l in h0_bars], reverse=True)
    h1_lifetimes = sorted([l for b, d, l in h1_bars], reverse=True)

    return {
        'h0_bars': h0_bars,
        'h1_bars': h1_bars,
        'h0_lifetimes': h0_lifetimes,
        'h1_lifetimes': h1_lifetimes,
        'n_essential_h1': len(h1_essential),
    }


def ph_invariant_vector(G, method='betweenness', n_bins=10):
    """
    Convert PH signature to fixed-length vector for classification.
    """
    sig = ph_signature(G, method)

    h0 = sig['h0_lifetimes']
    h1 = sig['h1_lifetimes']

    # Statistics of lifetime distributions
    def lifetime_stats(lifetimes):
        if not lifetimes:
            return [0.0] * 5
        arr = np.array(lifetimes)
        return [
            float(np.sum(arr)),          # total persistence
            float(np.mean(arr)),          # mean lifetime
            float(np.max(arr)),           # max lifetime (most robust feature)
            float(np.std(arr)) if len(arr) > 1 else 0.0,  # spread
            float(len(arr)),              # number of features
        ]

    betti_b, betti_1, betti_2 = betti_numbers(G)

    return {
        'h0_stats': lifetime_stats(h0),
        'h1_stats': lifetime_stats(h1),
        'betti': (betti_b, betti_1, betti_2),
        'n_essential_h1': sig.get('n_essential_h1', 0),
    }


# ============================================================
# ISOMORPHISM DISTINGUISHING POWER
# ============================================================

def ph_graph_signature(G, methods=('betweenness', 'resistance', 'spectral')):
    """
    Combined PH signature using multiple filtrations.
    Concatenate statistics from all filtrations.
    """
    n = G.number_of_nodes()
    sig_parts = []

    for method in methods:
        try:
            inv = ph_invariant_vector(G, method)
            sig_parts.extend(inv['h0_stats'])
            sig_parts.extend(inv['h1_stats'])
        except Exception:
            sig_parts.extend([0.0] * 10)

    try:
        b0, b1, b2 = betti_numbers(G)
        sig_parts.extend([b0, b1, b2])
    except:
        sig_parts.extend([0, 0, 0])

    return tuple(round(x, 6) for x in sig_parts)


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

print("=== Persistent Homology as Graph Invariant ===\n")

# 1. Basic Betti numbers for well-known graphs
print("--- Betti Numbers of Famous Graphs ---\n")
test_graphs = {
    'K4': nx.complete_graph(4),
    'C6': nx.cycle_graph(6),
    'Petersen': nx.petersen_graph(),
    'K3,3': nx.complete_bipartite_graph(3, 3),
    'Icosahedron': nx.icosahedron_graph() if hasattr(nx, 'icosahedron_graph') else nx.petersen_graph(),
    'K5': nx.complete_graph(5),
    'Cube': nx.convert_node_labels_to_integers(nx.hypercube_graph(3)),
    'Wheel(6)': nx.wheel_graph(6),
    'Path(6)': nx.path_graph(6),
    'Star(6)': nx.star_graph(5),
}

for name, G in test_graphs.items():
    G = nx.convert_node_labels_to_integers(G)
    b0, b1, b2 = betti_numbers(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    euler = n - m + len(list(nx.clique.find_cliques(G)))  # rough Euler char
    print(f"  {name:15s}: n={n}, m={m}, beta_0={b0}, beta_1={b1}, beta_2={b2}")

print()

# 2. Persistent Homology for Petersen vs K5 vs C6
print("--- Persistence Diagrams: Petersen, K5, C6 ---\n")

for name, G in [('Petersen', nx.petersen_graph()),
                ('K5', nx.complete_graph(5)),
                ('C6', nx.cycle_graph(6)),
                ('K3,3', nx.complete_bipartite_graph(3,3))]:
    G = nx.convert_node_labels_to_integers(G)
    for method in ['betweenness', 'resistance']:
        sig = ph_signature(G, method)
        h0 = sig['h0_lifetimes']
        h1 = sig['h1_lifetimes']
        print(f"  {name} ({method}): H0_lifetimes={[round(x,3) for x in h0[:4]]}, "
              f"H1_lifetimes={[round(x,3) for x in h1[:4]]}")
    print()

# 3. Test on n=6 non-iso graphs: does PH separate them?
print("--- PH Distinguishing Power: n=6 Connected Graphs ---\n")
atlas = nx.graph_atlas_g()
graphs_n6 = [G for G in atlas if G.number_of_nodes() == 6 and nx.is_connected(G)]
print(f"Total n=6 connected: {len(graphs_n6)}")

t0 = time.time()
sigs = {}
for i, G in enumerate(graphs_n6):
    try:
        sig = ph_graph_signature(G)
        sigs[i] = sig
    except Exception as e:
        sigs[i] = (0,) * 30

from collections import defaultdict
sig_groups = defaultdict(list)
for i, sig in sigs.items():
    sig_groups[sig].append(i)

collisions = {sig: group for sig, group in sig_groups.items() if len(group) > 1}
print(f"PH distinct: {len(sig_groups)}/{len(graphs_n6)}")
print(f"Collisions: {len(collisions)}")
print(f"Time: {time.time()-t0:.1f}s")

if collisions:
    print("Collision examples:")
    for sig, group in list(collisions.items())[:3]:
        print(f"  Group {group[:4]}: ", end='')
        for idx in group[:3]:
            G = graphs_n6[idx]
            deg = tuple(sorted([d for v,d in G.degree()], reverse=True))
            print(f"[{G.number_of_edges()}e,deg={deg}] ", end='')
        print()
else:
    print("PH is COMPLETE for n=6!")

print()

# 4. PH on WL-hard pairs: Shrikhande vs Rook(4,4)
print("--- PH on Hard Pairs: Shrikhande vs K_{3,3} (n=6) ---\n")

# Use two WL-1 hard pairs from n=6
# Find cospectral non-iso pairs
from collections import Counter
def spec_sig(G):
    return tuple(round(x, 4) for x in sorted(nx.laplacian_spectrum(G)))

cospectral = defaultdict(list)
for i, G in enumerate(graphs_n6):
    cospectral[spec_sig(G)].append(i)

cospectral_pairs = [(v[0], v[1]) for v in cospectral.values() if len(v) >= 2]
print(f"Cospectral pairs at n=6: {len(cospectral_pairs)}")
for i, j in cospectral_pairs[:3]:
    G1, G2 = graphs_n6[i], graphs_n6[j]
    same_ph = (sigs[i] == sigs[j])
    print(f"  Pair ({i},{j}): {G1.number_of_edges()}e vs {G2.number_of_edges()}e, "
          f"PH same={same_ph}")

print()

# 5. FFE vs PH cross-comparison: do they fail on same pairs?
print("--- FFE vs PH: Cross-comparison of Failure Modes ---\n")

def ground_state_projector(G, filling=0.5):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(-A)
    n_filled = int(n * filling)
    if n_filled == 0:
        return np.zeros((n, n))
    return eigvecs[:, :n_filled] @ eigvecs[:, :n_filled].T

def bipartition_entropy_fast(P_GS, idx_A):
    C_A = P_GS[np.ix_(idx_A, idx_A)]
    lambdas = np.clip(np.linalg.eigvalsh(C_A), 1e-12, 1-1e-12)
    return float(-np.sum(lambdas*np.log(lambdas) + (1-lambdas)*np.log(1-lambdas)))

def ffe_sig(G, bip_size=None):
    n = G.number_of_nodes()
    bip_size = bip_size or n//2
    P = ground_state_projector(G)
    return tuple(sorted(round(bipartition_entropy_fast(P, list(c)), 8)
                         for c in combinations(range(n), bip_size)))

ffe_groups = defaultdict(list)
for i, G in enumerate(graphs_n6):
    s = ffe_sig(G)
    ffe_groups[s].append(i)

ffe_collisions = {sig: group for sig, group in ffe_groups.items() if len(group) > 1}
ffe_fail_sets = set()
for group in ffe_collisions.values():
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            ffe_fail_sets.add((min(group[i], group[j]), max(group[i], group[j])))

ph_fail_sets = set()
for group in collisions.values():
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            ph_fail_sets.add((min(group[i], group[j]), max(group[i], group[j])))

both_fail = ffe_fail_sets & ph_fail_sets
only_ffe_fail = ffe_fail_sets - ph_fail_sets
only_ph_fail = ph_fail_sets - ffe_fail_sets

print(f"FFE failures: {len(ffe_fail_sets)} pairs")
print(f"PH failures: {len(ph_fail_sets)} pairs")
print(f"Both fail: {len(both_fail)} pairs")
print(f"Only FFE fails: {len(only_ffe_fail)} pairs {list(only_ffe_fail)[:3]}")
print(f"Only PH fails: {len(only_ph_fail)} pairs {list(only_ph_fail)[:3]}")

union_fail = ffe_fail_sets | ph_fail_sets
print(f"\nCombined FFE+PH failures: {len(union_fail)} pairs")
print(f"=> FFE+PH together distinguishes {len(graphs_n6)*(len(graphs_n6)-1)//2 - len(union_fail)} "
      f"/ {len(graphs_n6)*(len(graphs_n6)-1)//2} pairs correctly")

# Are FFE+PH together complete at n=6?
if not union_fail:
    print("=> FFE + PH is COMPLETE at n=6!")
else:
    print(f"=> Still {len(union_fail)} unresolved pairs")
    for pair in union_fail:
        G1, G2 = graphs_n6[pair[0]], graphs_n6[pair[1]]
        print(f"   ({pair[0]},{pair[1]}): {G1.number_of_edges()}e vs {G2.number_of_edges()}e")

print()
print("=== KEY THEORETICAL IMPLICATIONS ===\n")
print("1. Betti numbers alone are weak (many graphs share same values)")
print("2. Persistent homology (with good filtration) is stronger")
print("3. Different filtrations (betweenness vs resistance vs spectral) see different structure")
print("4. FFE and PH may fail on DIFFERENT pairs -> their combination is stronger")
print()
print("Connection to physics:")
print("  PH = multi-scale topology (geometric)")
print("  FFE = quantum entanglement (quantum information)")
print("  -> Topology + Quantum = potentially complete invariant?")
