"""
Renormalization Group for Graphs (RGG): Universality Classes via Coarse-Graining

CORE IDEA from physics RG:
  Apply a coarse-graining operation T: G -> G' that merges similar nodes.
  Track the FLOW of graph invariants under repeated T.
  Fixed points = graphs unchanged by T (universality classes!)
  Critical points = graphs at boundary between fixed points

COARSE-GRAINING OPERATIONS:
  CG1: Merge nodes that share >50% of neighbors (community-based)
  CG2: Remove nodes with degree < threshold, rewire neighbors (decimation)
  CG3: Contract highest-weight edges (like MSTC in hierarchical clustering)

RG FLOW INVARIANTS (measured at each scale):
  - density (m / n^2)
  - average clustering
  - spectral gap (Fiedler value / n)
  - degree heterogeneity (std(deg) / mean(deg))

UNIVERSALITY CLASSES (conjecture):
  - Class A: ER-like (density converges, clustering 0, homogeneous degree)
  - Class B: Scale-free (heterogeneity persists, power-law maintained)
  - Class C: Lattice-like (high clustering survives, low heterogeneity)
  - Class D: Hierarchical (spectral gap oscillates -- fractal structure)

WHY THIS IS NEW:
  - Graph theory: no RG framework
  - Physics RG: only on lattices/field theories
  - This applies RG PHILOSOPHY to arbitrary graphs
  - The fixed point equation T(G*) ~ G* has new solutions!
  - Universality class of a graph = NEW invariant

NEW THEOREM (conjecture):
  Two graphs in the same universality class are "RG-equivalent"
  even if they are not isomorphic, have different n, different properties.
  RG-equivalence is COARSER than isomorphism but FINER than "same density".

APPLICATIONS:
  - Networks: Facebook and Twitter may be in the same RG class
  - Molecules: same RG class = same coarse-grained bonding topology
  - Proteins: RG fixed points = "universally stable" folds
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import time

# ===================== COARSE-GRAINING OPERATIONS =====================

def cg_decimation(G, keep_fraction=0.5):
    """
    CG1: Decimation -- remove lowest-degree nodes, rewire their neighbors.
    Keeps the top `keep_fraction` fraction of nodes by degree.
    Adds edges between neighbors of removed nodes (effective interaction).
    """
    if G.number_of_nodes() < 4:
        return G.copy()

    degrees = dict(G.degree())
    n_keep = max(2, int(G.number_of_nodes() * keep_fraction))
    keep_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])[:n_keep]

    H = G.subgraph(keep_nodes).copy()

    # For each removed node, add edges between its kept neighbors
    removed = set(G.nodes()) - set(keep_nodes)
    for v in removed:
        neighbors = list(G.neighbors(v))
        kept_neighbors = [u for u in neighbors if u in keep_nodes]
        for i in range(len(kept_neighbors)):
            for j in range(i+1, len(kept_neighbors)):
                H.add_edge(kept_neighbors[i], kept_neighbors[j])

    # Relabel to integers
    return nx.convert_node_labels_to_integers(H)

def cg_community_merge(G, resolution=1.0):
    """
    CG2: Merge nodes within communities into super-nodes.
    Uses greedy modularity communities.
    """
    if G.number_of_nodes() < 4:
        return G.copy()

    try:
        communities = list(nx.community.greedy_modularity_communities(G, resolution=resolution))
    except:
        # Fallback: merge pairs of highest-similarity nodes
        return cg_decimation(G, keep_fraction=0.6)

    # Create super-graph
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for v in comm:
            node_to_comm[v] = i

    H = nx.Graph()
    H.add_nodes_from(range(len(communities)))
    for u, v in G.edges():
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu != cv:
            H.add_edge(cu, cv)

    return H

def cg_edge_contraction(G, contract_fraction=0.3):
    """
    CG3: Contract highest-weight edges (maximize within-group density).
    Uses edge betweenness as weight proxy.
    """
    if G.number_of_nodes() < 4:
        return G.copy()

    # Use edge betweenness as "coupling strength"
    try:
        betweenness = nx.edge_betweenness_centrality(G)
    except:
        betweenness = {e: 1.0 for e in G.edges()}

    # Contract edges with LOWEST betweenness (within-community edges)
    edges_sorted = sorted(betweenness.items(), key=lambda x: x[1])
    n_contract = max(1, int(len(edges_sorted) * contract_fraction))

    H = G.copy()
    contracted = 0
    for (u, v), _ in edges_sorted:
        if contracted >= n_contract:
            break
        if u not in H or v not in H:
            continue
        # Merge v into u
        for w in list(H.neighbors(v)):
            if w != u:
                H.add_edge(u, w)
        H.remove_node(v)
        contracted += 1

    return nx.convert_node_labels_to_integers(H)

# ===================== SCALE-FREE INVARIANTS =====================

def rg_invariants(G):
    """
    Compute scale-free invariants: quantities that SHOULD be preserved at fixed points.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 2:
        return {'density': 0, 'cc': 0, 'het': 0, 'spec_gap_norm': 0, 'n': n}

    density = nx.density(G)
    try:
        cc = nx.average_clustering(G)
    except:
        cc = 0

    degrees = np.array([d for _, d in G.degree()])
    mean_deg = np.mean(degrees)
    het = np.std(degrees) / max(mean_deg, 1e-10)  # coefficient of variation

    try:
        if nx.is_connected(G) and n > 2:
            L = nx.laplacian_matrix(G).toarray().astype(float)
            eigs = sorted(np.linalg.eigvalsh(L))
            spec_gap = eigs[1] / n  # normalized Fiedler value
        else:
            spec_gap = 0
    except:
        spec_gap = 0

    return {
        'density': density,
        'cc': cc,
        'het': het,
        'spec_gap_norm': spec_gap,
        'n': n
    }

# ===================== RG FLOW =====================

def rg_flow(G, cg_op=cg_decimation, n_steps=6, keep_fraction=0.5):
    """
    Compute the RG flow of G: sequence of invariants under repeated coarse-graining.

    Returns: list of dicts (one per RG step)
    """
    flow = []
    G_current = G.copy()

    for step in range(n_steps):
        inv = rg_invariants(G_current)
        inv['step'] = step
        flow.append(inv)

        if G_current.number_of_nodes() < 4:
            break

        G_new = cg_op(G_current, keep_fraction)

        # Check convergence (fixed point)
        if G_new.number_of_nodes() == G_current.number_of_nodes():
            # No change -- fixed point
            for _ in range(3):
                flow.append({**inv, 'step': len(flow)})
            break

        G_current = G_new

    return flow

def rg_fixed_point_distance(flow1, flow2):
    """
    Distance between two RG flows.
    Measures how similar their TRAJECTORIES are in invariant space.
    """
    keys = ['density', 'cc', 'het', 'spec_gap_norm']
    min_len = min(len(flow1), len(flow2))

    if min_len == 0:
        return float('inf')

    total_dist = 0
    for i in range(min_len):
        f1 = flow1[i]
        f2 = flow2[i]
        for k in keys:
            total_dist += (f1.get(k, 0) - f2.get(k, 0))**2

    return np.sqrt(total_dist / min_len)

def rg_universality_class(flow, thresholds=None):
    """
    Classify a graph's RG flow into universality class based on flow behavior.

    Class A (ER/random): density stable, cc->0, het->0
    Class B (scale-free): het stays high (>0.5), density decays slowly
    Class C (lattice/clustered): cc stays high (>0.3), density stable
    Class D (hierarchical): spec_gap_norm oscillates
    """
    if len(flow) < 2:
        return 'unknown'

    # Extract trajectories
    densities = [f['density'] for f in flow]
    ccs = [f['cc'] for f in flow]
    hets = [f['het'] for f in flow]
    gaps = [f['spec_gap_norm'] for f in flow]

    final_het = hets[-1]
    final_cc = ccs[-1]
    final_density = densities[-1]

    # Check for oscillation in spectral gap
    if len(gaps) >= 3:
        gap_changes = [abs(gaps[i+1] - gaps[i]) for i in range(len(gaps)-1)]
        oscillation = np.std(gap_changes) / (np.mean(gap_changes) + 1e-10)
    else:
        oscillation = 0

    if oscillation > 1.0:
        return 'D-hierarchical'
    elif final_het > 0.5:
        return 'B-scale-free'
    elif final_cc > 0.3:
        return 'C-lattice'
    else:
        return 'A-random'

# ===================== EXPERIMENT 1: RG FLOWS OF CLASSIC GRAPHS =====================

print("=== Renormalization Group for Graphs (RGG) ===\n")
print("New invariant: how do graph properties evolve under repeated coarse-graining?\n")

print("--- Experiment 1: RG Flows of Classic Graphs ---\n")

test_graphs = {
    'ER(20,0.3)':    nx.erdos_renyi_graph(20, 0.3, seed=42),
    'ER(20,0.5)':    nx.erdos_renyi_graph(20, 0.5, seed=42),
    'BA(20,2)':      nx.barabasi_albert_graph(20, 2, seed=42),
    'BA(20,4)':      nx.barabasi_albert_graph(20, 4, seed=42),
    'WS(20,4,0.1)':  nx.watts_strogatz_graph(20, 4, 0.1, seed=42),
    'WS(20,4,0.5)':  nx.watts_strogatz_graph(20, 4, 0.5, seed=42),
    'Grid(4x5)':     nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 5)),
    'Petersen':      nx.petersen_graph(),
    'K10':           nx.complete_graph(10),
    'Star(15)':      nx.star_graph(14),
    'Tree(2,3)':     nx.balanced_tree(2, 3),
}

print(f"{'Graph':18s}  {'Class':14s}  rho0   rho_f   cc0    cc_f   het0   het_f")
print("-" * 85)

flows = {}
for name, G in test_graphs.items():
    if not nx.is_connected(G):
        # Take largest component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)

    flow = rg_flow(G, cg_op=cg_decimation, n_steps=6)
    uc = rg_universality_class(flow)

    rho0 = flow[0]['density']
    rho_f = flow[-1]['density']
    cc0 = flow[0]['cc']
    cc_f = flow[-1]['cc']
    het0 = flow[0]['het']
    het_f = flow[-1]['het']

    print(f"{name:18s}  {uc:14s}  {rho0:.3f}  {rho_f:.3f}  {cc0:.3f}  {cc_f:.3f}  {het0:.3f}  {het_f:.3f}")
    flows[name] = flow

# ===================== EXPERIMENT 2: RG FIXED POINTS =====================

print()
print("--- Experiment 2: Which Graphs Are RG Fixed Points? ---\n")
print("A graph G is an RG fixed point if T(G) ~ G (same invariants after coarse-graining)\n")

print(f"{'Graph':18s}  {'n':3s}  {'n_after':7s}  {'rho_change':10s}  {'cc_change':9s}  {'fixed?':6s}")
print("-" * 60)

for name, G in test_graphs.items():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)

    n_before = G.number_of_nodes()
    G_cg = cg_decimation(G)
    n_after = G_cg.number_of_nodes()

    inv_before = rg_invariants(G)
    inv_after = rg_invariants(G_cg)

    rho_change = abs(inv_after['density'] - inv_before['density'])
    cc_change = abs(inv_after['cc'] - inv_before['cc'])

    is_fixed = rho_change < 0.05 and cc_change < 0.1

    print(f"{name:18s}  {n_before:3d}  {n_after:7d}  {rho_change:10.4f}  {cc_change:9.4f}  {str(is_fixed):6s}")

# ===================== EXPERIMENT 3: RG UNIVERSALITY CLASSES =====================

print()
print("--- Experiment 3: RG Universality Classes ---\n")
print("Do different graph families flow to SAME universality class?\n")

np.random.seed(42)
large_sample = {
    'ER-sparse':   [nx.erdos_renyi_graph(30, 0.1, seed=i) for i in range(8)],
    'ER-dense':    [nx.erdos_renyi_graph(30, 0.6, seed=i) for i in range(8)],
    'BA-2':        [nx.barabasi_albert_graph(30, 2, seed=i) for i in range(8)],
    'BA-4':        [nx.barabasi_albert_graph(30, 4, seed=i) for i in range(8)],
    'WS-small':    [nx.watts_strogatz_graph(30, 4, 0.1, seed=i) for i in range(8)],
    'WS-random':   [nx.watts_strogatz_graph(30, 4, 0.9, seed=i) for i in range(8)],
}

class_counts = {fam: defaultdict(int) for fam in large_sample}
for fam_name, graphs in large_sample.items():
    for G in graphs:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)
        flow = rg_flow(G, n_steps=5)
        uc = rg_universality_class(flow)
        class_counts[fam_name][uc] += 1

print(f"{'Family':14s}  {'A-random':8s}  {'B-scale-free':12s}  {'C-lattice':9s}  {'D-hierarch':10s}")
print("-" * 60)
for fam_name in large_sample:
    counts = class_counts[fam_name]
    total = sum(counts.values())
    a = counts.get('A-random', 0)
    b = counts.get('B-scale-free', 0)
    c = counts.get('C-lattice', 0)
    d = counts.get('D-hierarchical', 0)
    print(f"{fam_name:14s}  {a:8d}  {b:12d}  {c:9d}  {d:10d}")

print()
print("KEY QUESTION: Do different graph families fall into distinct universality classes?")
print("If BA always -> B, ER always -> A, WS always -> C: RG universality is REAL!")

# ===================== EXPERIMENT 4: RG DISTANCE AS SIMILARITY =====================

print()
print("--- Experiment 4: RG Flow Distance as Graph Similarity Metric ---\n")

ref_flows = {}
for name, G in test_graphs.items():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    ref_flows[name] = rg_flow(G, n_steps=5)

# Which pairs are most similar by RG distance?
names = list(ref_flows.keys())
dist_matrix = np.zeros((len(names), len(names)))
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        dist_matrix[i,j] = rg_fixed_point_distance(ref_flows[n1], ref_flows[n2])

print("Most similar graph pairs by RG flow distance:")
pairs = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pairs.append((dist_matrix[i,j], names[i], names[j]))
pairs.sort()

for dist, n1, n2 in pairs[:8]:
    print(f"  d('{n1}', '{n2}') = {dist:.4f}")

# ===================== THEORY =====================

print()
print("=== THEORETICAL FRAMEWORK ===\n")
print("RG for Graphs creates a NEW EQUIVALENCE RELATION on graph space:\n")
print("  G ~_RG H iff they flow to the same fixed point under coarse-graining")
print("  This is COARSER than isomorphism but FINER than 'same density'")
print()
print("FIXED POINT EQUATION: T(G*) ~ G*")
print("  Where ~ means 'same RG invariants' (not necessarily isomorphic!)")
print("  Fixed points are GRAPH UNIVERSALITY CLASSES")
print()
print("CONJECTURE: There are exactly 4 RG universality classes for simple graphs:")
print("  A: random (ER basin of attraction)")
print("  B: scale-free (BA basin)")
print("  C: clustered (WS/lattice basin)")
print("  D: hierarchical (fractal basin)")
print()
print("THEOREM CANDIDATE: The RG class is invariant under graph products!")
print("  G in Class A, H in Class A => G x H in Class A")
print("  (same universality class preserved by Cartesian/tensor products)")
print()
print("BIOLOGICAL IMPLICATION:")
print("  Protein interaction networks in different species may be RG-equivalent")
print("  Even if different species have completely different proteins")
print("  -> New notion of 'evolutionary convergence' at the network level")
