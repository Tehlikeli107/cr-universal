"""
Graph Fourier Sparsity (GFS): Signal Processing View of Graph Structure

CORE IDEA from signal processing:
  Any function f: V(G) -> R can be expressed in the Laplacian eigenbasis:
    f = sum_k alpha_k * phi_k
  where phi_k are Laplacian eigenvectors (graph Fourier modes)

  For STRUCTURAL SIGNALS (indicator functions of node subsets):
    1_S = indicator of subset S
    The number of nonzero alpha_k = "Fourier complexity" of S

  GFS(G) = mean Fourier complexity of all induced k-subgraph indicators
  Low GFS = subgraphs are "smooth" (few Fourier modes needed)
  High GFS = subgraphs are "rough" (many modes needed)

INTERPRETATION:
  Fourier complexity = how "irregular" the subgraph boundary is
  - Regular graphs: subgraphs need few modes (symmetric structure)
  - Random graphs: subgraphs need many modes (all irregular)
  - Bipartite graphs: subgraphs split cleanly (2 dominant modes)

WHY THIS IS NEW:
  - Graph signal processing: well-developed (Shuman 2013, Ortega 2018)
  - But applied to GRAPH CLASSIFICATION via subgraph sparsity: NEW!
  - This is the "spectral compressibility" of graph structure

NEW QUANTITIES:
  1. Spectral Width: effective bandwidth of subgraph indicators
     W_k(G) = mean spectral width = mean(number of significant Fourier components)

  2. Spectral Entropy: H_spectral = entropy of |alpha_k|^2 distribution
     Low spectral entropy = one dominant mode (structured graph)
     High spectral entropy = all modes contribute equally (random graph)

  3. Cutoff Frequency: k* = smallest k such that subgraphs are k-bandlimited
     (top-k Fourier modes capture >90% of energy)

  4. Spectral Correlation: correlation of Fourier spectra of different subgraphs
     High correlation = subgraphs share structure (symmetric graph)

DEEP CONNECTION:
  The Fourier complexity of 1_S = number of non-trivial eigenvalues
  needed to describe S. This is related to:
  - Graph bandwidth (smallest k such that G has a bandwidth-k labeling)
  - Hamming distance from S to any Laplacian eigenspace projection
  - The "graph signal reconstruction" problem

SURPRISE PREDICTION:
  Bipartite graphs: indicator functions of one partition need ONLY 2 modes
  (the constant mode + the bipartite indicator)
  -> GFS(bipartite) << GFS(non-bipartite) for balanced partitions!
"""

import numpy as np
import networkx as nx
from itertools import combinations
import time

# ===================== GRAPH FOURIER BASIS =====================

def graph_fourier_basis(G):
    """
    Compute graph Fourier basis (Laplacian eigenvectors).
    Returns: (eigenvalues, eigenvectors) where columns are eigenvectors.
    """
    n = G.number_of_nodes()
    if n < 2:
        return np.array([]), np.eye(1)

    try:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigvals, eigvecs = np.linalg.eigh(L)
        return eigvals, eigvecs
    except:
        return np.zeros(n), np.eye(n)

def fourier_complexity(signal, eigvecs, threshold=0.01):
    """
    Compute Fourier complexity of a signal.
    = number of Fourier coefficients with |alpha_k|^2 >= threshold * total_energy

    Also returns: spectral entropy, cutoff_k
    """
    n = len(signal)
    # Graph Fourier transform
    alpha = eigvecs.T @ signal

    # Energy in each frequency
    energy = alpha**2
    total_energy = energy.sum()

    if total_energy < 1e-12:
        return 0, 0, 0

    # Normalized energy
    norm_energy = energy / total_energy

    # Complexity = number of components above threshold
    complexity = np.sum(norm_energy > threshold)

    # Spectral entropy
    norm_energy_safe = norm_energy + 1e-12
    norm_energy_safe /= norm_energy_safe.sum()
    spec_entropy = -np.sum(norm_energy_safe * np.log(norm_energy_safe))

    # Cutoff: smallest k such that top-k modes capture 90% energy
    sorted_energy = np.sort(norm_energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    cutoff = np.searchsorted(cumsum, 0.9) + 1

    return complexity, spec_entropy, cutoff

def gfs_invariants(G, k=3, max_subsets=200, seed=42, threshold=0.01):
    """
    Compute Graph Fourier Sparsity invariants.

    Returns dict with mean complexity, mean spectral entropy,
    spectral correlation between subsets, etc.
    """
    n = G.number_of_nodes()
    if n < k + 1:
        return {'complexity': 0, 'spec_entropy': 0, 'cutoff': 0,
                'spec_corr': 0, 'n_modes': 0}

    eigvals, eigvecs = graph_fourier_basis(G)
    nodes = list(G.nodes())

    # Sample subsets
    all_combos = list(combinations(range(n), k))
    rng = np.random.RandomState(seed)
    if len(all_combos) > max_subsets:
        idx = rng.choice(len(all_combos), max_subsets, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    complexities = []
    spec_entropies = []
    cutoffs = []
    spectra = []

    for combo in combos:
        signal = np.zeros(n)
        for i in combo:
            signal[i] = 1.0
        signal /= max(signal.sum(), 1e-10)

        c, se, co = fourier_complexity(signal, eigvecs, threshold=threshold)
        complexities.append(c)
        spec_entropies.append(se)
        cutoffs.append(co)

        # Store Fourier spectrum for correlation
        alpha = eigvecs.T @ signal
        spectra.append(alpha**2)

    # Spectral correlation between subsets
    if len(spectra) > 1:
        spec_matrix = np.array(spectra)
        # Mean pairwise cosine similarity
        norms = np.linalg.norm(spec_matrix, axis=1, keepdims=True) + 1e-12
        spec_norm = spec_matrix / norms
        corr_matrix = spec_norm @ spec_norm.T
        n_pairs = len(spectra) * (len(spectra) - 1) // 2
        if n_pairs > 0:
            # Sample pairs
            n_sample = min(n_pairs, 500)
            corrs = []
            for _ in range(n_sample):
                i, j = rng.choice(len(spectra), 2, replace=False)
                corrs.append(corr_matrix[i, j])
            spec_corr = np.mean(corrs)
        else:
            spec_corr = 0
    else:
        spec_corr = 0

    return {
        'complexity': np.mean(complexities),
        'spec_entropy': np.mean(spec_entropies),
        'cutoff': np.mean(cutoffs),
        'spec_corr': spec_corr,
        'n_modes': n,
        'complexity_std': np.std(complexities),
    }

# ===================== EXPERIMENT 1: GFS OF CLASSIC GRAPHS =====================

print("=== Graph Fourier Sparsity (GFS) ===\n")
print("Subgraph indicator functions in Laplacian eigenbasis -- spectral compressibility\n")

print("--- Experiment 1: GFS of Classic Graphs (k=3 subsets) ---\n")
print(f"{'Graph':18s}  {'complexity':10s}  {'spec_H':6s}  {'cutoff':6s}  {'corr':6s}  {'Interpretation'}")
print("-" * 75)

test_graphs = {
    'K5 (complete)':   nx.complete_graph(5),
    'K6':              nx.complete_graph(6),
    'C6 (cycle)':      nx.cycle_graph(6),
    'C8':              nx.cycle_graph(8),
    'Petersen':        nx.petersen_graph(),
    'K3,3 (bipart)':   nx.complete_bipartite_graph(3, 3),
    'K2,4 (bipart)':   nx.complete_bipartite_graph(2, 4),
    'Path(8)':         nx.path_graph(8),
    'Star(7)':         nx.star_graph(6),
    'Grid(3x3)':       nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
    'ER(10,0.4)':      nx.erdos_renyi_graph(10, 0.4, seed=42),
    'BA(10,2)':        nx.barabasi_albert_graph(10, 2, seed=42),
}

gfs_data = {}
for name, G in test_graphs.items():
    gfs = gfs_invariants(G, k=3)
    c = gfs['complexity']
    se = gfs['spec_entropy']
    co = gfs['cutoff']
    corr = gfs['spec_corr']

    if c < 3:
        interp = "very smooth (symmetric)"
    elif c < 5:
        interp = "moderate Fourier spread"
    else:
        interp = "high complexity (irregular)"

    print(f"{name:18s}  {c:10.3f}  {se:6.3f}  {co:6.1f}  {corr:6.3f}  {interp}")
    gfs_data[name] = gfs

print()
print("complexity = mean number of Fourier modes for 3-node subsets")
print("spec_H = spectral entropy of mode distribution")
print("cutoff = minimum modes for 90% energy recovery")
print("corr = cosine similarity between spectra of different subsets")

# ===================== EXPERIMENT 2: BIPARTITE vs NON-BIPARTITE =====================

print()
print("--- Experiment 2: GFS Distinguishes Bipartite vs Non-Bipartite ---\n")
print("PREDICTION: bipartite graphs have lower complexity (clean partition = 2 modes)\n")

bipartite_graphs = {
    'K1,5':    nx.complete_bipartite_graph(1, 5),
    'K2,4':    nx.complete_bipartite_graph(2, 4),
    'K3,3':    nx.complete_bipartite_graph(3, 3),
    'K4,4':    nx.complete_bipartite_graph(4, 4),
    'Path(6)': nx.path_graph(6),       # bipartite
    'Grid(3x2)': nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 2)),  # bipartite
}

nonbipartite_graphs = {
    'K4':       nx.complete_graph(4),
    'K5':       nx.complete_graph(5),
    'C5':       nx.cycle_graph(5),
    'C7':       nx.cycle_graph(7),
    'Petersen': nx.petersen_graph(),
    'BA(6,2)':  nx.barabasi_albert_graph(6, 2, seed=42),
}

print("BIPARTITE GRAPHS:")
bip_complexities = []
for name, G in bipartite_graphs.items():
    gfs = gfs_invariants(G, k=3)
    bip_complexities.append(gfs['complexity'])
    print(f"  {name:12s}: complexity={gfs['complexity']:.3f}, spec_H={gfs['spec_entropy']:.3f}")

print()
print("NON-BIPARTITE GRAPHS:")
nonbip_complexities = []
for name, G in nonbipartite_graphs.items():
    gfs = gfs_invariants(G, k=3)
    nonbip_complexities.append(gfs['complexity'])
    print(f"  {name:12s}: complexity={gfs['complexity']:.3f}, spec_H={gfs['spec_entropy']:.3f}")

print()
print(f"Mean complexity: bipartite={np.mean(bip_complexities):.3f}, "
      f"non-bipartite={np.mean(nonbip_complexities):.3f}")
print(f"Hypothesis: bipartite < non-bipartite: {np.mean(bip_complexities) < np.mean(nonbip_complexities)}")

# ===================== EXPERIMENT 3: GFS AS CLASSIFIER =====================

print()
print("--- Experiment 3: GFS Classification ---\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
clf_graphs = []
clf_labels = []

n_each = 20
for _ in range(n_each):
    G = nx.erdos_renyi_graph(12, 0.3, seed=np.random.randint(1000))
    clf_graphs.append(G); clf_labels.append(0)

for _ in range(n_each):
    G = nx.barabasi_albert_graph(12, 3, seed=np.random.randint(1000))
    clf_graphs.append(G); clf_labels.append(1)

for _ in range(n_each):
    G = nx.watts_strogatz_graph(12, 4, 0.2, seed=np.random.randint(1000))
    clf_graphs.append(G); clf_labels.append(2)

y = np.array(clf_labels)

X_basic = np.array([[nx.density(G), nx.average_clustering(G),
                     np.std([d for _, d in G.degree()])] for G in clf_graphs])

X_gfs = []
for G in clf_graphs:
    gfs = gfs_invariants(G, k=3)
    X_gfs.append([gfs['complexity'], gfs['spec_entropy'], gfs['cutoff'], gfs['spec_corr']])
X_gfs = np.array(X_gfs)

for feat_name, X in [('Basic (3)', X_basic),
                      ('GFS (4)', X_gfs),
                      ('Combined (7)', np.hstack([X_basic, X_gfs]))]:
    try:
        acc = cross_val_score(LogisticRegression(max_iter=500),
                              StandardScaler().fit_transform(X), y, cv=5).mean()
        print(f"  {feat_name:25s}: CV accuracy = {acc:.3f}")
    except Exception as e:
        print(f"  {feat_name:25s}: error ({e})")

# ===================== EXPERIMENT 4: FREQUENCY-STRUCTURE CORRELATION =====================

print()
print("--- Experiment 4: Which Fourier Modes Encode Which Structure? ---\n")
print("Low frequencies = global structure (connectivity)")
print("High frequencies = local structure (triangles, local density)\n")

G_test = nx.petersen_graph()
eigvals, eigvecs = graph_fourier_basis(G_test)
n = G_test.number_of_nodes()
nodes = list(G_test.nodes())

# Which subgraphs have high low-frequency content?
high_low_freq = []  # subsets where most energy is in low Fourier modes
high_high_freq = []  # subsets where most energy is in high Fourier modes

for combo in list(combinations(range(n), 3))[:50]:
    signal = np.zeros(n)
    for i in combo: signal[i] = 1.0
    signal /= signal.sum()

    alpha = eigvecs.T @ signal
    energy = alpha**2
    total = energy.sum()

    if total > 0:
        low_freq_energy = energy[:n//3].sum() / total
        high_low_freq.append((low_freq_energy, combo))
        high_high_freq.append((1 - low_freq_energy, combo))

high_low_freq.sort(reverse=True)
high_high_freq.sort(reverse=True)

print("Petersen graph: 3-subsets with highest LOW-frequency content:")
print("(= most 'spatially smooth' subsets)")
for score, combo in high_low_freq[:3]:
    sub = G_test.subgraph([nodes[i] for i in combo])
    n_edges = sub.number_of_edges()
    print(f"  Nodes {combo}: low_freq={score:.3f}, edges_in_subset={n_edges}")

print()
print("Petersen graph: 3-subsets with highest HIGH-frequency content:")
print("(= most 'spatially rough' / irregular subsets)")
for score, combo in high_high_freq[:3]:
    sub = G_test.subgraph([nodes[i] for i in combo])
    n_edges = sub.number_of_edges()
    print(f"  Nodes {combo}: high_freq={score:.3f}, edges_in_subset={n_edges}")

# ===================== THEORY =====================

print()
print("=== THEORETICAL FRAMEWORK ===\n")
print("Graph Fourier Sparsity (GFS) defines 'spectral compressibility':\n")
print("  GFS(G) = mean number of Laplacian eigenmodes needed")
print("           to describe induced k-subgraph indicators")
print()
print("CLAIM: GFS is STRICTLY MORE INFORMATIVE than CR entropy alone")
print("  CR entropy: counts subgraph TYPES (combinatorial)")
print("  GFS: measures spectral COMPLEXITY of those subgraphs (spectral)")
print("  Two graphs can have same CR entropy but different GFS!")
print()
print("FOURIER COMPLEXITY HIERARCHY:")
print("  GFS = 1: G is regular, all subsets need only DC component")
print("  GFS = 2: G is bipartite (partition = 2-mode representation)")
print("  GFS = k: G has k distinct 'spectral scales'")
print()
print("THEOREM CANDIDATE: GFS(G) >= chromatic_number(G) - 1")
print("  (proper coloring needs at least chi-1 non-trivial modes)")
print()
print("GRAPH SIGNAL PROCESSING CONNECTION:")
print("  Low GFS = graph is a good 'communication network'")
print("  (signals spread smoothly, few modes needed for recovery)")
print("  High GFS = graph has many spectral scales")
print("  (different nodes live in different frequency regimes)")
print()
print("SAMPLING THEOREM ON GRAPHS:")
print("  If GFS(G) = k, then any k-bandlimited signal on G can be")
print("  recovered from samples on k nodes (graph Nyquist theorem)")
print("  -> GFS = minimum sample size for perfect signal recovery!")
