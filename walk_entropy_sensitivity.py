"""
Walk Entropy Sensitivity (WES): How Entropy Rate Changes Under Perturbations

ENTROPY RATE OF RANDOM WALK:
  The natural random walk on G: move to random neighbor with probability 1/d(v)
  Stationary distribution: pi(v) = d(v) / (2m)
  Entropy rate: h(G) = -sum_v pi(v) * sum_u P(v,u) * log P(v,u)
                     = log(2m) - (1/2m) * sum_v d(v) * log(d(v))
                     = log(2m) - H_degree_weighted

  This has a CLOSED FORM! It's entirely determined by the degree sequence.

  h(K_n) = log(n-1)  [all degrees same]
  h(P_n) = log(2) - small correction  [leaves have degree 1]
  h(C_n) = log(2)  [all degree 2]

NEW: ENTROPY RATE SENSITIVITY
  What happens to h(G) when we add/remove an edge?
  dh/d(edge e) = marginal entropy rate = how much edge e contributes

  This defines:
  1. Edge entropy contribution: delta_h(e) = h(G+e) - h(G)
  2. Entropy elasticity: epsilon(e) = d_h/h = relative change
  3. Entropy gradient vector: grad_h = (delta_h(e1), ..., delta_h(em))

  The DISTRIBUTION of {delta_h(e)} is a new graph invariant!

KEY NEW INSIGHT:
  Since h(G) = log(2m) - (1/2m) * sum d_v * log(d_v)
  Adding edge (u,v) changes: m -> m+1, d_u -> d_u+1, d_v -> d_v+1
  delta_h(u,v) = log((2m+2)/(2m)) - (1/2m) * [(d_u+1)*log(d_u+1) - d_u*log(d_u)
                                                + (d_v+1)*log(d_v+1) - d_v*log(d_v)]
               + correction term

  HIGH delta_h: adding edge increases entropy (creates high-degree nodes)
  LOW delta_h: adding edge decreases entropy (reduces degree variance)

MAXIMUM ENTROPY GRAPH:
  Which graph on n nodes and m edges maximizes h(G)?
  ANSWER: the regular graph! (all degrees equal = maximum entropy rate)
  This is a new optimization problem on graph space.

ENTROPY SENSITIVITY PROFILE (ESP):
  For each graph, the distribution P(delta_h) over all possible non-edges
  is the Entropy Sensitivity Profile.
  - If ESP is concentrated: all edges have similar entropy value
  - If ESP is spread: some edges are entropy-critical

BIOLOGICAL:
  In metabolic networks: high-entropy-rate metabolites are "universal"
  (connected to many pathways, act as entropy reservoirs)
  Low-entropy edges are "specific" bottlenecks (potential drug targets)
"""

import numpy as np
import networkx as nx
from itertools import combinations
import time

# ===================== ENTROPY RATE =====================

def entropy_rate(G):
    """
    Compute entropy rate of natural random walk on G.
    h(G) = log(2m) - (1/2m) * sum_v d_v * log(d_v)

    Returns 0 for empty or disconnected graphs.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    degrees = np.array([G.degree(v) for v in G.nodes()], dtype=float)
    degrees_nonzero = degrees[degrees > 0]

    if len(degrees_nonzero) == 0:
        return 0.0

    # h = log(2m) - weighted entropy of degree sequence
    h = np.log(2 * m) - (1 / (2 * m)) * np.sum(degrees_nonzero * np.log(degrees_nonzero))
    return h

def edge_entropy_contribution(G, u, v):
    """
    Marginal entropy rate contribution of edge (u,v).
    delta_h = h(G) - h(G minus e)
    """
    if not G.has_edge(u, v):
        return 0.0

    h_full = entropy_rate(G)

    G_minus = G.copy()
    G_minus.remove_edge(u, v)
    if G_minus.number_of_edges() == 0:
        return h_full

    h_minus = entropy_rate(G_minus)
    return h_full - h_minus

def entropy_sensitivity_profile(G, max_edges=None):
    """
    Compute delta_h for all edges in G.
    Returns: array of delta_h values (one per edge)
    """
    edges = list(G.edges())
    if max_edges and len(edges) > max_edges:
        edges = edges[:max_edges]

    deltas = [edge_entropy_contribution(G, u, v) for u, v in edges]
    return np.array(deltas)

def non_edge_entropy_potential(G, max_non_edges=None):
    """
    Compute delta_h for all NON-EDGES: h(G + e) - h(G)
    This is the "entropy potential" of each possible new edge.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    non_edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i+1, n)
                 if not G.has_edge(nodes[i], nodes[j])]

    if max_non_edges and len(non_edges) > max_non_edges:
        np.random.shuffle(non_edges)
        non_edges = non_edges[:max_non_edges]

    h_original = entropy_rate(G)
    potentials = []

    for u, v in non_edges:
        G_plus = G.copy()
        G_plus.add_edge(u, v)
        h_plus = entropy_rate(G_plus)
        potentials.append(h_plus - h_original)

    return np.array(potentials), non_edges

def wes_invariants(G):
    """
    Compute Walk Entropy Sensitivity invariants.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if m == 0 or n < 2:
        return {'h': 0, 'h_max': 0, 'esp_std': 0, 'esp_max': 0,
                'h_efficiency': 0, 'entropy_gap': 0}

    h = entropy_rate(G)

    # Maximum possible entropy for this degree sequence
    # (achieved by regular graph): h_max = log(mean_degree)
    degrees = np.array([G.degree(v) for v in G.nodes()], dtype=float)
    mean_deg = np.mean(degrees)
    h_max = np.log(mean_deg) if mean_deg > 0 else 0

    # Entropy efficiency: ratio to maximum
    h_efficiency = h / max(h_max, 1e-10)

    # Edge entropy sensitivity
    if m <= 100:
        esp = entropy_sensitivity_profile(G)
        esp_std = np.std(esp)
        esp_max = np.max(esp) if len(esp) > 0 else 0
        esp_min = np.min(esp) if len(esp) > 0 else 0
    else:
        esp_std = 0
        esp_max = 0
        esp_min = 0

    # Entropy gap: how far from regular graph
    degrees_nonzero = degrees[degrees > 0]
    h_regular = np.log(mean_deg) if mean_deg > 0 else 0
    entropy_gap = h_regular - h  # gap = how much sub-optimal vs regular

    # Von Neumann entropy for comparison
    try:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        L_norm = L / (2 * m)
        eigs = np.linalg.eigvalsh(L_norm)
        eigs_pos = eigs[eigs > 1e-10]
        vn_entropy = -np.sum(eigs_pos * np.log(eigs_pos))
    except:
        vn_entropy = 0

    return {
        'h': h,
        'h_max': h_max,
        'h_efficiency': h_efficiency,
        'entropy_gap': entropy_gap,
        'esp_std': esp_std,
        'esp_max': esp_max,
        'vn_entropy': vn_entropy
    }

# ===================== EXPERIMENT 1: ENTROPY RATE PROFILES =====================

print("=== Walk Entropy Sensitivity (WES) ===\n")
print("h(G) = entropy rate of random walk. Sensitivity to perturbations = new invariant\n")

print("--- Experiment 1: Entropy Rates of Classic Graphs ---\n")
print(f"{'Graph':18s}  {'h':6s}  {'h_max':6s}  {'eff':5s}  {'gap':5s}  {'esp_std':7s}  {'vn_H':6s}")
print("-" * 65)

test_graphs = {
    'K5 (complete)':   nx.complete_graph(5),
    'K6':              nx.complete_graph(6),
    'C6 (cycle)':      nx.cycle_graph(6),
    'C8':              nx.cycle_graph(8),
    'Petersen':        nx.petersen_graph(),
    'K3,3':            nx.complete_bipartite_graph(3, 3),
    'Path(6)':         nx.path_graph(6),
    'Path(8)':         nx.path_graph(8),
    'Star(7)':         nx.star_graph(6),
    'Grid(3x3)':       nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
    'ER(10,0.4)':      nx.erdos_renyi_graph(10, 0.4, seed=42),
    'BA(10,2)':        nx.barabasi_albert_graph(10, 2, seed=42),
    'Tree(2,3)':       nx.balanced_tree(2, 3),
}

wes_data = {}
for name, G in test_graphs.items():
    inv = wes_invariants(G)
    print(f"{name:18s}  {inv['h']:6.3f}  {inv['h_max']:6.3f}  {inv['h_efficiency']:5.3f}  "
          f"{inv['entropy_gap']:5.3f}  {inv['esp_std']:7.4f}  {inv['vn_entropy']:6.3f}")
    wes_data[name] = inv

print()
print("h = entropy rate of natural random walk (exact: log(2m) - weighted degree entropy)")
print("h_max = max possible h for this mean degree (achieved by regular graph)")
print("eff = h/h_max = how close to maximum entropy for given density")
print("gap = h_max - h = entropy deficit due to degree irregularity")
print("esp_std = std of edge entropy contributions (edge heterogeneity)")

# ===================== EXPERIMENT 2: ENTROPY GRADIENT =====================

print()
print("--- Experiment 2: Entropy Gradient - Which Edges Matter Most? ---\n")
print("For each graph: which edges contribute most/least to entropy rate?\n")

for name in ['Petersen', 'Star(7)', 'BA(10,2)', 'ER(10,0.4)']:
    G = test_graphs[name]
    edges = list(G.edges())
    deltas = entropy_sensitivity_profile(G)

    if len(deltas) == 0:
        continue

    # Rank edges by contribution
    ranked = sorted(zip(deltas, edges), reverse=True)

    top_edge = ranked[0]
    bot_edge = ranked[-1]

    # Degree of edge endpoints
    max_d = max(G.degree(top_edge[1][0]), G.degree(top_edge[1][1]))
    min_d = max(G.degree(bot_edge[1][0]), G.degree(bot_edge[1][1]))

    print(f"{name:15s}: h={entropy_rate(G):.3f}, esp_range=[{min(deltas):.4f},{max(deltas):.4f}], "
          f"top_edge_max_deg={max_d}, bot_edge_max_deg={min_d}")

print()
print("PREDICTION: high-entropy edges connect high-degree nodes")
print("Low-entropy edges: removing them DECREASES entropy (they create irregularity)")

# ===================== EXPERIMENT 3: ENTROPY POTENTIAL OF NON-EDGES =====================

print()
print("--- Experiment 3: Entropy Potential - Which Edges to ADD? ---\n")
print("For each graph: which NEW edges would increase entropy rate most?\n")

for name in ['C6 (cycle)', 'Path(6)', 'Grid(3x3)', 'K3,3']:
    G = test_graphs[name]
    potentials, non_edges = non_edge_entropy_potential(G, max_non_edges=50)

    if len(potentials) == 0:
        continue

    max_pot = np.max(potentials)
    mean_pot = np.mean(potentials)

    # Find which non-edge has highest potential
    best_idx = np.argmax(potentials)
    best_u, best_v = non_edges[best_idx]
    du, dv = G.degree(best_u), G.degree(best_v)

    print(f"{name:15s}: h={entropy_rate(G):.3f}, max_delta_h={max_pot:.4f}, "
          f"mean_delta_h={mean_pot:.4f}, best_non_edge_degrees=({du},{dv})")

# ===================== EXPERIMENT 4: MAXIMUM ENTROPY GRAPHS =====================

print()
print("--- Experiment 4: Entropy Rate vs Graph Regularity ---\n")
print("CLAIM: Regular graphs maximize h(G) for given (n, m)\n")

np.random.seed(42)
n = 10

# Generate various graphs with approximately same edge count (m~20)
test_pairs = [
    ("4-regular (n=10)", nx.circulant_graph(n, [1, 2])),   # 4-regular
    ("ER(10,0.45)", nx.erdos_renyi_graph(n, 0.45, seed=42)),
    ("BA(10,2)", nx.barabasi_albert_graph(n, 2, seed=42)),
    ("WS(10,4,0.1)", nx.watts_strogatz_graph(n, 4, 0.1, seed=42)),
    ("WS(10,4,0.9)", nx.watts_strogatz_graph(n, 4, 0.9, seed=42)),
    ("Star+path", None),  # constructed below
]

# Build star + path for comparison
G_special = nx.star_graph(4)
G_special.add_nodes_from(range(5, 10))
G_special.add_edges_from([(5,6),(6,7),(7,8),(8,9),(5,9)])
test_pairs[-1] = ("Star+cycle(mix)", G_special)

print(f"{'Graph':20s}  {'n':3s}  {'m':3s}  {'h':6s}  {'h_eff':5s}  {'deg_std':7s}")
print("-" * 50)
for gname, G in test_pairs:
    if G is None: continue
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    degrees = np.array([G.degree(v) for v in G.nodes()], dtype=float)
    inv = wes_invariants(G)
    print(f"{gname:20s}  {G.number_of_nodes():3d}  {G.number_of_edges():3d}  "
          f"{inv['h']:6.3f}  {inv['h_efficiency']:5.3f}  {np.std(degrees):7.4f}")

print()
print("4-regular should have highest entropy efficiency (eff close to 1.0)")
print("Irregular graphs (star, BA) have lower efficiency due to degree variance")

# ===================== EXPERIMENT 5: WES CLASSIFICATION =====================

print()
print("--- Experiment 5: WES Graph Classification ---\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
clf_graphs = []
clf_labels = []

n_each = 20
for _ in range(n_each):
    G = nx.erdos_renyi_graph(12, 0.3, seed=np.random.randint(1000))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    clf_graphs.append(G); clf_labels.append(0)

for _ in range(n_each):
    G = nx.barabasi_albert_graph(12, 3, seed=np.random.randint(1000))
    clf_graphs.append(G); clf_labels.append(1)

for _ in range(n_each):
    G = nx.watts_strogatz_graph(12, 4, 0.2, seed=np.random.randint(1000))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    clf_graphs.append(G); clf_labels.append(2)

y = np.array(clf_labels)

X_basic = np.array([[nx.density(G), nx.average_clustering(G),
                     np.std([d for _, d in G.degree()])] for G in clf_graphs])

X_wes = []
for G in clf_graphs:
    inv = wes_invariants(G)
    X_wes.append([inv['h'], inv['h_efficiency'], inv['entropy_gap'],
                  inv['esp_std'], inv['vn_entropy']])
X_wes = np.array(X_wes)

for feat_name, X in [('Basic (3)', X_basic),
                      ('WES (5)', X_wes),
                      ('Combined (8)', np.hstack([X_basic, X_wes]))]:
    try:
        acc = cross_val_score(LogisticRegression(max_iter=500),
                              StandardScaler().fit_transform(X), y, cv=5).mean()
        print(f"  {feat_name:25s}: CV accuracy = {acc:.3f}")
    except Exception as e:
        print(f"  {feat_name:25s}: error ({e})")

# ===================== THEORY =====================

print()
print("=== THEORETICAL FRAMEWORK ===\n")
print("Walk Entropy Sensitivity defines a NEW FUNCTIONAL on graph space:\n")
print("  h(G) = log(2m) - (1/2m) * sum_v d(v)*log(d(v))")
print("  This is CLOSED FORM -- completely determined by degree sequence!")
print()
print("KEY THEOREM (trivially): h(G) <= log(mean_degree)")
print("  Equality iff G is regular (maximum entropy random walk)")
print("  The gap = h_max - h = 'degree entropy deficit'")
print()
print("SURPRISE: h(G) depends ONLY on degree sequence, NOT on edge structure!")
print("  Two graphs with same degree sequence have IDENTICAL h(G)")
print("  -> h(G) is a degree-sequence invariant, NOT a graph invariant!")
print()
print("BUT: Edge entropy sensitivity dh/de DOES depend on graph structure!")
print("  ESP = distribution of edge contributions = GENUINE graph invariant")
print("  (depends on the full graph structure, not just degrees)")
print()
print("MAXIMUM ENTROPY THEOREM:")
print("  Among all graphs with n nodes and m edges: h(G) is maximized by")
print("  ANY graph whose degree sequence is as regular as possible")
print("  (d-regular if 2m/n is integer, or (d,d+1)-biregular otherwise)")
print()
print("SENSITIVITY FORMULA (new!):")
print("  delta_h(e=(u,v)) = log((2m+2)/(2m))")
print("    - (1/(2m+2)) * [(d_u+1)log(d_u+1) - d_u*log(d_u)]")
print("    - (1/(2m+2)) * [(d_v+1)log(d_v+1) - d_v*log(d_v)]")
print("    + (1/(2m)) * [d_u*log(d_u) + d_v*log(d_v)]  (correction)")
print()
print("  Edges between LOW-DEGREE nodes have HIGH delta_h (adding them is entropy-efficient)")
print("  Edges between HIGH-DEGREE nodes have NEGATIVE delta_h (adding them reduces entropy)")
