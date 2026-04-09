"""
k_min=2 Threshold: What Minimum c_node Makes k_min=2?

QUESTION: For unlabeled graphs (c_edge=1 only), what is the minimum
number of node types c_node such that k=2 subgraph distribution
uniquely identifies all non-isomorphic labeled graphs?

If k_min=2 at c_node=X, then: knowing the PAIR FREQUENCY DISTRIBUTION
(for each node-type pair, how many edges exist between them) uniquely
determines the graph.

CONJECTURE: k_min <= 2 iff c_node >= sqrt(n) or c_node >= f(n)

THEORETICAL INSIGHT:
  k=2 fingerprint with c_node types = a c_node x c_node matrix:
    M[a][b] = number of (node type-a) - (node type-b) edges
    D[a] = count of type-a nodes

  This is the "typed degree sequence". It UNIQUELY determines:
  - How many nodes of each type
  - How many edges between each type pair

  It FAILS when two graphs have the same typed degree sequences
  but different connectivity patterns (e.g., isomers at the type level).

  With c_node >= n, all nodes have unique types -> k_min=1 (trivially determined by edges)
  With c_node = 2 (binary labels): this is a SIGNED GRAPH, k_min=?
"""
import numpy as np
from itertools import combinations, permutations
from collections import Counter
import random

def canonical_node_labeled_graph(n, node_labels, edges):
    """edges: set of (i,j) pairs i<j"""
    def make_adj(perm):
        mat = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(node_labels[perm[i]])
                else:
                    row.append(1 if (min(perm[i],perm[j]), max(perm[i],perm[j])) in edges else 0)
            mat.append(tuple(row))
        return tuple(mat)

    best = None
    for perm in permutations(range(n)):
        adj = make_adj(perm)
        if best is None or adj < best:
            best = adj
    return best

def cr_fingerprint_k2_node(n, node_labels, edges):
    """k=2 fingerprint: pair (label_u, label_v, edge?) distribution."""
    c = Counter()
    for i in range(n):
        for j in range(i+1, n):
            la, lb = node_labels[i], node_labels[j]
            key_min, key_max = min(la, lb), max(la, lb)
            has_edge = (i, j) in edges or (j, i) in edges
            c[(key_min, key_max, int(has_edge))] += 1
    return c

def generate_noniso_node_labeled(n, c_node, max_graphs=400):
    """Generate non-iso node-labeled unweighted graphs."""
    all_edge_pairs = [(i,j) for i in range(n) for j in range(i+1, n)]
    n_e = len(all_edge_pairs)
    canonical_to_graph = {}

    n_node_configs = c_node ** n
    n_edge_configs = 2 ** n_e
    n_total = n_node_configs * n_edge_configs

    if n_total <= 30000:
        for node_mask in range(n_node_configs):
            nl = []
            tmp = node_mask
            for i in range(n):
                nl.append((tmp % c_node) + 1)
                tmp //= c_node

            for edge_mask in range(n_edge_configs):
                edges = set()
                for bit, (u,v) in enumerate(all_edge_pairs):
                    if edge_mask & (1 << bit):
                        edges.add((u, v))

                canon = canonical_node_labeled_graph(n, nl, edges)
                if canon not in canonical_to_graph:
                    canonical_to_graph[canon] = (nl[:], edges, canon)
    else:
        for _ in range(max_graphs * 20):
            nl = [random.randint(1, c_node) for _ in range(n)]
            edges = set()
            for u, v in all_edge_pairs:
                if random.random() < 0.5:
                    edges.add((u, v))

            canon = canonical_node_labeled_graph(n, nl, edges)
            if canon not in canonical_to_graph:
                canonical_to_graph[canon] = (nl[:], edges, canon)
            if len(canonical_to_graph) >= max_graphs:
                break

    return list(canonical_to_graph.values())

def find_kmin_k2_k3(graphs, n):
    """Test k=2 and k=3 only."""
    # Test k=2
    fps2 = {}
    col2 = 0
    for nl, edges, canon in graphs:
        fp = cr_fingerprint_k2_node(n, nl, edges)
        h = frozenset(fp.items())
        if h in fps2 and fps2[h] != canon:
            col2 += 1
        fps2[h] = canon

    if col2 == 0:
        return 2

    # Test k=3 (slow but needed)
    from itertools import permutations as perms
    def cr_fp_k3(n, nl, edges):
        c = Counter()
        for sub in combinations(range(n), 3):
            nl2 = sorted(sub)
            k = len(nl2)
            def make_adj(perm):
                mat = []
                for i in range(k):
                    row = []
                    for j in range(k):
                        if i == j:
                            row.append(nl[nl2[perm[i]]])
                        else:
                            has_e = (min(nl2[perm[i]], nl2[perm[j]]), max(nl2[perm[i]], nl2[perm[j]])) in edges
                            row.append(1 if has_e else 0)
                    mat.append(tuple(row))
                return tuple(mat)
            best = None
            for perm in perms(range(k)):
                adj = make_adj(perm)
                if best is None or adj < best:
                    best = adj
            c[best] += 1
        return c

    fps3 = {}
    col3 = 0
    for nl, edges, canon in graphs:
        fp = cr_fp_k3(n, nl, edges)
        h = frozenset(fp.items())
        if h in fps3 and fps3[h] != canon:
            col3 += 1
        fps3[h] = canon

    if col3 == 0:
        return 3
    return 4  # Need k>=4

print("=== k_min=2 Threshold: Minimum c_node for k_min=2 ===\n")
print("Testing k=2 (typed pair distribution) and k=3 for node-labeled unweighted graphs\n")
random.seed(42)
np.random.seed(42)

# Test for various n and c_node
results = {}
for n in [4, 5, 6, 7]:
    results[n] = {}
    print(f"n={n}:", flush=True)
    max_c = min(n, 10)
    for c_node in range(1, max_c + 1):
        print(f"  c_node={c_node}:", end='', flush=True)
        graphs = generate_noniso_node_labeled(n, c_node, max_graphs=400)
        print(f" {len(graphs)} graphs", end='', flush=True)
        kmin = find_kmin_k2_k3(graphs, n)
        results[n][c_node] = kmin
        print(f" -> k_min={kmin}", flush=True)
        if kmin <= 2:
            print(f"  THRESHOLD: c_node={c_node} -> k_min=2 for n={n}!")
            break
    print()

print("=== THRESHOLD TABLE ===\n")
print(f"{'n':>4s}  {'min c_node for k_min=2':>22s}")
print("-" * 30)
for n, res in sorted(results.items()):
    thresh = None
    for c, kmin in sorted(res.items()):
        if kmin <= 2:
            thresh = c
            break
    thresh_str = str(thresh) if thresh else ">tested"
    print(f"  {n:2d}  {thresh_str:>22s}")

print()
print("PATTERN: Does threshold = 2? 3? sqrt(n)? n/2?")
