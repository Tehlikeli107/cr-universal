"""
Label-k_min Conjecture: Extended to Node+Edge Labels

EXTENSION: Test with BOTH node labels AND edge labels:
  - c_node: number of distinct node label types (atom types in molecular = ~10)
  - c_edge: number of distinct edge label types (bond types = 4)
  - Combined label entropy higher -> k_min should be even smaller

Compare:
  1. Untyped (c_node=1, c_edge=1): known k_min ~ n-3
  2. Edge-only (c_node=1, c_edge=4): k_min=3 (from cr_label_kmin.py)
  3. Node-only (c_node=10, c_edge=1): k_min=?
  4. Molecular-like (c_node=10, c_edge=4): k_min=? (could be 2!)
  5. High-entropy (c_node=20, c_edge=4): k_min=?

PREDICTION: Node labels alone should bring k_min to 3 or below.
Node+Edge labels might bring k_min to 2 (just need pairs of atoms!).

This would explain:
  - Why CR k=2 uniquely identifies 58.8% of ESOL molecules
  - Why CR k=3 identifies 75.4% more (needs 3-subgraph patterns)
  - The "long tail" at k>=4 is molecules with unusual composition
"""
import numpy as np
from itertools import combinations, permutations
from collections import Counter
import random

def canonical_labeled_graph_node_edge(n, node_labels, edge_labels):
    """
    Canonical form with both node and edge labels.
    node_labels: dict {node_id: label (int)}
    edge_labels: dict {(i,j): color (int)} for i<j
    """
    def make_adj(perm):
        mat = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(node_labels.get(perm[i], 1))
                else:
                    pi, pj = perm[i], perm[j]
                    key = (min(pi,pj), max(pi,pj))
                    row.append(edge_labels.get(key, 0))
            mat.append(tuple(row))
        return tuple(mat)

    best = None
    for perm in permutations(range(n)):
        adj = make_adj(perm)
        if best is None or adj < best:
            best = adj
    return best

def induced_sub_canonical_node_edge(n_global, node_labels, edge_labels, subset):
    """Canonical form of induced k-subgraph."""
    k = len(subset)
    sl = sorted(subset)
    sub_node = {i: node_labels.get(sl[i], 1) for i in range(k)}
    sub_edge = {}
    for i in range(k):
        for j in range(i+1, k):
            key = (min(sl[i], sl[j]), max(sl[i], sl[j]))
            c = edge_labels.get(key, 0)
            if c > 0:
                sub_edge[(i, j)] = c
    return canonical_labeled_graph_node_edge(k, sub_node, sub_edge)

def cr_fingerprint_node_edge(n, node_labels, edge_labels, k):
    """CR k-subgraph fingerprint."""
    c = Counter()
    for sub in combinations(range(n), k):
        canon = induced_sub_canonical_node_edge(n, node_labels, edge_labels, sub)
        c[canon] += 1
    return c

def generate_graphs_node_edge(n, c_node, c_edge, max_graphs=300):
    """Generate non-isomorphic graphs with node+edge labels."""
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    n_edges = len(all_edges)
    canonical_to_graph = {}

    n_configs_node = c_node ** n
    n_configs_edge = (c_edge + 1) ** n_edges
    n_total = n_configs_node * n_configs_edge

    if n_total <= 20000:
        # Exact enumeration
        for node_mask in range(n_configs_node):
            node_labels = {}
            tmp = node_mask
            for i in range(n):
                node_labels[i] = (tmp % c_node) + 1
                tmp //= c_node

            for edge_mask in range(n_configs_edge):
                edge_labels = {}
                tmp = edge_mask
                for bit, (u,v) in enumerate(all_edges):
                    color = tmp % (c_edge + 1)
                    if color > 0:
                        edge_labels[(u,v)] = color
                    tmp //= (c_edge + 1)

                canon = canonical_labeled_graph_node_edge(n, node_labels, edge_labels)
                if canon not in canonical_to_graph:
                    canonical_to_graph[canon] = (node_labels.copy(), edge_labels.copy(), canon)
    else:
        # Random sampling
        for _ in range(max_graphs * 20):
            node_labels = {i: random.randint(1, c_node) for i in range(n)}
            edge_labels = {}
            for u, v in all_edges:
                color = random.randint(0, c_edge)
                if color > 0:
                    edge_labels[(u,v)] = color

            canon = canonical_labeled_graph_node_edge(n, node_labels, edge_labels)
            if canon not in canonical_to_graph:
                canonical_to_graph[canon] = (node_labels.copy(), edge_labels.copy(), canon)
            if len(canonical_to_graph) >= max_graphs:
                break

    return list(canonical_to_graph.values())

def find_kmin_node_edge(graphs, n, k_max=5):
    """Find k_min for node+edge labeled graphs."""
    for k in range(2, k_max+1):
        fps = {}
        collisions = 0
        for node_labels, edge_labels, canon in graphs:
            fp = cr_fingerprint_node_edge(n, node_labels, edge_labels, k)
            fp_hash = frozenset(fp.items())
            if fp_hash in fps and fps[fp_hash] != canon:
                collisions += 1
            fps[fp_hash] = canon
        if collisions == 0:
            return k
    return k_max + 1

print("=== Label-k_min Extended: Node+Edge Labels ===\n")
print("Testing: k_min with both node labels AND edge labels\n")

random.seed(42)
np.random.seed(42)

# Test configurations
configs = [
    (1, 1, "untyped"),
    (1, 4, "edge-4"),
    (10, 1, "node-10"),
    (4, 4, "node4+edge4"),
    (10, 4, "molecular (node10+edge4)"),
]

results = {}
for n in [5, 6, 7]:
    results[n] = {}
    print(f"n={n} nodes:")
    for c_node, c_edge, label in configs:
        print(f"  {label} (c_node={c_node}, c_edge={c_edge}):", flush=True)
        graphs = generate_graphs_node_edge(n, c_node, c_edge, max_graphs=400)
        n_graphs = len(graphs)
        print(f"    {n_graphs} non-iso graphs", flush=True)
        if n_graphs < 2:
            continue
        k_max = min(n-1, 5)
        kmin = find_kmin_node_edge(graphs, n, k_max=k_max)
        results[n][label] = kmin
        print(f"    k_min = {kmin}", flush=True)
    print()

print("=== RESULTS TABLE ===\n")
header = f"{'n':>3s}  " + "  ".join(f"{lbl[:12]:>12s}" for _, _, lbl in configs)
print(header)
print("-" * (5 + 14 * len(configs)))
for n in [5, 6, 7]:
    row = f"  {n:1d}  "
    for _, _, lbl in configs:
        val = results.get(n, {}).get(lbl, '?')
        row += f"  {str(val):>12s}"
    print(row)

print()
print("EXTENDED CONJECTURE:")
print("  k_min depends on TOTAL label entropy:")
print("  H = n*log(c_node) + |E|*log(c_edge+1)")
print()
print("  Molecular graphs: H = n*log(10) + |E|*log(5)")
print("  -> Maximum entropy per subgraph type")
print("  -> k_min=2 possible for high-entropy graphs!")
print()
print("If node-10 alone gives k_min<=3: node labels do most of the work.")
print("If molecular (node10+edge4) gives k_min=2: COMPOSITION ALONE is sufficient!")
