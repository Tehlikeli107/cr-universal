"""
FFE Completeness Test: Does Free Fermion Entanglement Solve Graph Isomorphism?

QUESTION: For all non-isomorphic graphs G, H on n nodes:
  Does there exist a bipartition A such that S_G(A) != S_H(A)?

If YES: FFE is a COMPLETE graph invariant (stronger than WL hierarchy!)
        -> Quantum information theory SOLVES graph isomorphism!

If NO: Find the smallest counterexample pair. What is it?

COMPUTATIONAL PLAN:
  1. Enumerate all non-isomorphic graphs on n=4,5,6,7 nodes
  2. Compute FFE signature = sorted vector of S(A) for ALL bipartitions
  3. Check if any two non-iso graphs have identical FFE signatures
  4. Report: is FFE complete? If not, find counterexample.

THEORY CONTEXT:
  WL-1: fails on Shrikhande vs Rook(4,4)
  WL-k: fails on CFI graphs
  Counting Revolution: fails on quantum isomorphic graphs (n>=120)
  FFE: ???

  FFE uses quantum AMPLITUDE rather than just counts.
  Could potentially separate more pairs than WL hierarchy.
  But if it fails on ANY pair, we have a new hard pair to study!

Note: bipartition size = n//2 (balanced bipartitions only)
For n=7: all C(7,3) = 35 bipartitions
For n=8: all C(8,4) = 70 bipartitions
"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
import time

def ground_state_projector(G, filling=0.5):
    """Compute ground state projector for free fermions on G."""
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros((n, n))

    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(-A)  # H = -A

    n_filled = int(n * filling)
    if n_filled == 0:
        return np.zeros((n, n))

    P_GS = eigvecs[:, :n_filled] @ eigvecs[:, :n_filled].T
    return P_GS

def bipartition_entropy_fast(P_GS, idx_A):
    """Compute S(A) given projector and index set."""
    C_A = P_GS[np.ix_(idx_A, idx_A)]
    lambdas = np.linalg.eigvalsh(C_A)
    lambdas = np.clip(lambdas, 1e-12, 1 - 1e-12)
    S = -np.sum(lambdas * np.log(lambdas) + (1 - lambdas) * np.log(1 - lambdas))
    return float(S)

def ffe_signature(G, bip_size=None):
    """
    Compute FFE signature: sorted tuple of S(A) for ALL bipartitions of size bip_size.
    This is a graph invariant (isomorphism-invariant).
    """
    n = G.number_of_nodes()
    if n < 2:
        return (0.0,)

    if bip_size is None:
        bip_size = n // 2

    P_GS = ground_state_projector(G)
    nodes = list(G.nodes())

    entropies = []
    for combo in combinations(range(n), bip_size):
        S = bipartition_entropy_fast(P_GS, list(combo))
        entropies.append(round(S, 8))  # round to avoid floating point noise

    return tuple(sorted(entropies))

def graphs_on_n_nodes(n, connected_only=True):
    """
    Generate all non-isomorphic graphs on n nodes.
    Uses networkx graph atlas for small n.
    """
    all_graphs = []

    # For n <= 7, use networkx graph_atlas_g() or enumerate
    if n <= 7:
        try:
            atlas = nx.graph_atlas_g()
            for G in atlas:
                if G.number_of_nodes() == n:
                    if connected_only and not nx.is_connected(G):
                        continue
                    all_graphs.append(G)
        except:
            pass

    if not all_graphs:
        # Fallback: enumerate by hand for small n
        from itertools import product as iproduct
        m_max = n * (n-1) // 2
        edges_list = list(combinations(range(n), 2))
        seen = []
        for mask in range(2**m_max):
            edges = [edges_list[i] for i in range(m_max) if (mask >> i) & 1]
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            if connected_only and not nx.is_connected(G):
                continue
            # Check isomorphism with existing
            is_new = True
            for G2 in seen:
                if nx.is_isomorphic(G, G2):
                    is_new = False
                    break
            if is_new:
                seen.append(G)
                all_graphs.append(G)

    return all_graphs

# ===================== MAIN TEST =====================

print("=== FFE Completeness Test ===\n")
print("Question: Does Free Fermion Entanglement separate ALL non-isomorphic graphs?\n")

t0 = time.time()

for n in [4, 5, 6, 7]:
    print(f"\n--- n = {n} ---")
    t1 = time.time()

    graphs = graphs_on_n_nodes(n, connected_only=True)
    print(f"Connected non-iso graphs on n={n}: {len(graphs)}")

    if not graphs:
        print("  (No graphs found)")
        continue

    # Compute FFE signatures
    sigs = {}
    for i, G in enumerate(graphs):
        sig = ffe_signature(G)
        sigs[i] = sig

    # Find collisions
    sig_groups = defaultdict(list)
    for i, sig in sigs.items():
        sig_groups[sig].append(i)

    collisions = {sig: group for sig, group in sig_groups.items() if len(group) > 1}
    n_distinct = len(sig_groups)
    n_total = len(graphs)

    print(f"  FFE distinct: {n_distinct}/{n_total}")
    print(f"  Collisions (non-iso graphs with same FFE): {len(collisions)}")

    if collisions:
        print(f"  !!! FFE is INCOMPLETE for n={n} !!!")
        for sig, group in list(collisions.items())[:3]:
            print(f"\n  Collision group (size {len(group)}):")
            for idx in group[:4]:
                G = graphs[idx]
                deg_seq = tuple(sorted([G.degree(v) for v in G.nodes()], reverse=True))
                n_edges = G.number_of_edges()
                cc = nx.average_clustering(G)
                print(f"    Graph {idx}: {n_edges} edges, deg={deg_seq}, cc={cc:.3f}")
    else:
        print(f"  FFE is COMPLETE for n={n} (all graphs separated)")

    print(f"  Time: {time.time()-t1:.1f}s")

print(f"\n=== Total time: {time.time()-t0:.1f}s ===")

# ===================== COMPARISON WITH WL =====================

print("\n--- Comparison: WL-1 vs FFE on hard pairs ---\n")

# Known WL-1 hard pairs (cospectral)
def shrikhande_rook():
    """Shrikhande graph and Rook(4,4) = cospectral pair, WL-1 fails."""
    # Shrikhande graph: vertices = Z_4 x Z_4, adjacency = differ by (0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1)
    n = 16
    G1 = nx.Graph()
    G1.add_nodes_from(range(16))
    for i in range(4):
        for j in range(4):
            v = 4*i + j
            for di, dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1)]:
                u = 4*((i+di)%4) + (j+dj)%4
                G1.add_edge(v, u)

    # Rook(4,4) = K4 x K4 (tensor product)
    G2 = nx.rook_graph(4)

    return G1, G2

print("Shrikhande vs Rook(4,4): n=16, WL-1 FAILS")
print("Testing FFE (n=16 is large, using subset of bipartitions)...")

try:
    G_sh, G_rk = shrikhande_rook()
    n = 16

    P_sh = ground_state_projector(G_sh)
    P_rk = ground_state_projector(G_rk)

    nodes = list(range(n))
    np.random.seed(42)

    diffs = []
    for _ in range(50):
        combo = tuple(np.random.choice(n, n//2, replace=False))
        idx = list(combo)
        S_sh = bipartition_entropy_fast(P_sh, idx)
        S_rk = bipartition_entropy_fast(P_rk, idx)
        diffs.append(abs(S_sh - S_rk))

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    n_same = sum(d < 1e-6 for d in diffs)

    print(f"  50 random bipartitions: max|S_sh - S_rk| = {max_diff:.6f}")
    print(f"  Mean difference = {mean_diff:.6f}")
    print(f"  Same bipartitions: {n_same}/50")

    if max_diff < 1e-4:
        print("  -> FFE FAILS to distinguish Shrikhande vs Rook! (like WL-1)")
    else:
        print("  -> FFE SUCCEEDS where WL-1 fails!")

except Exception as e:
    print(f"  Error: {e}")

# ===================== HARD PAIR ANALYSIS =====================

print()
print("--- Testing n=8 Hard Pairs from Counting Revolution ---\n")
# These pairs are hard for k=5 induced subgraph counting but resolved at k=6

hard_pairs_n8 = [
    (3630, 4580),
    (7163, 7210),
]

try:
    atlas = nx.graph_atlas_g()
    atlas_n8 = [G for G in atlas if G.number_of_nodes() == 8 and nx.is_connected(G)]

    for g1_idx, g2_idx in hard_pairs_n8:
        # Find in atlas
        if g1_idx < len(atlas) and g2_idx < len(atlas):
            G1 = atlas[g1_idx]
            G2 = atlas[g2_idx]

            if G1.number_of_nodes() != 8 or G2.number_of_nodes() != 8:
                print(f"  Pair ({g1_idx},{g2_idx}): wrong n")
                continue

            sig1 = ffe_signature(G1, bip_size=4)
            sig2 = ffe_signature(G2, bip_size=4)

            same = (sig1 == sig2)
            same_exact = all(abs(s1-s2) < 1e-6 for s1, s2 in zip(sig1, sig2))

            print(f"  Pair ({g1_idx},{g2_idx}): FFE_same={same_exact}")
            if same_exact:
                print(f"    -> FFE FAILS on this hard pair!")
            else:
                print(f"    -> FFE SUCCEEDS on this hard pair!")
                diff = max(abs(s1-s2) for s1, s2 in zip(sig1, sig2))
                print(f"    max difference = {diff:.8f}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== THEORETICAL IMPLICATIONS ===\n")
print("If FFE is COMPLETE for all n: quantum information theory solves GI!")
print("If FFE is INCOMPLETE: the counterexample pair is a new hard instance")
print("  that might be related to quantum isomorphic graphs.")
print()
print("Connection to quantum isomorphism:")
print("  Two graphs are quantum isomorphic iff they have the same homomorphism")
print("  counts from all planar graphs (Mancinska-Roberson 2020)")
print("  FFE uses quantum STATES, not homomorphism counts.")
print("  -> They are DIFFERENT notions of quantum equivalence!")
