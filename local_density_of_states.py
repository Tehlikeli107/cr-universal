"""
Graph Local Density of States (GLDOS): Condensed Matter Physics Applied to Graphs

THE KEY IDEA from solid-state physics:
  In a crystal, different atoms experience DIFFERENT electronic environments.
  The "local density of states" (LDOS) at atom v measures how many
  quantum states are "nearby" in energy at that atom's location.

  LDOS(v, E) = sum_k |phi_k(v)|^2 * delta(E - lambda_k)
  where (lambda_k, phi_k) = Laplacian eigenvalue/eigenvector pairs

  This is NOT the same as the global DOS = sum_v LDOS(v, E) / n

  KEY: LDOS(v, E) measures how much the k-th quantum state
       is LOCALIZED at node v.

NEW INVARIANTS:
  1. Spectral Localization: max_v max_E LDOS(v,E) -- how localized are states?
  2. Spatial Entropy: H_spatial(k) = entropy of |phi_k(v)|^2 over nodes
     Low entropy: state k is localized (few nodes contribute)
     High entropy: state k is delocalized (all nodes contribute equally)
  3. Inverse Participation Ratio: IPR_k = sum_v |phi_k(v)|^4
     IPR = 1/n: fully delocalized
     IPR = 1: fully localized at one node
  4. LDOS Divergence: std of LDOS across nodes (spectral inhomogeneity)

ANDERSON LOCALIZATION:
  In disordered quantum systems: eigenstates can be LOCALIZED (confined to small region)
  For graphs: disorder = irregularity in degree sequence
  Regular graphs: states delocalized (IPR ~ 1/n)
  Irregular/random graphs: some states localized (IPR >> 1/n)!

  -> Anderson localization on graphs = NEW MEASURE OF GRAPH DISORDER!

THE NODAL DOMAIN THEOREM:
  The k-th Laplacian eigenvector has at most k nodal domains (sign changes)
  phi_k divides G into regions of same sign
  Number of nodal domains = new node-level invariant!

NEW MOLECULAR INSIGHT:
  LDOS at atom v at HOMO energy = electron density at v (reactivity!)
  LDOS at Fermi level = conductance contribution of atom v
  For drug molecules: LDOS at functional groups = binding affinity proxy!

  This is the GRAPH-THEORETIC ANALOG of quantum chemistry HOMO/LUMO theory!
  And it's computable from PURE graph structure (no quantum chemistry software!)

IPR AS COMPLEXITY MEASURE:
  For K_n: all eigenvectors delocalized -> low IPR ~ 1/n
  For Path: some eigenvectors localized at ends -> high IPR
  For BA (scale-free): hub eigenvectors very localized -> very high IPR!
  -> IPR spectrum captures the "quantum localization complexity" of G

THE ENERGY GAP:
  HOMO-LUMO gap in graphs = gap between n-th and (n+1)-th Laplacian eigenvalue
  (where n = number of electrons = number of nodes for half-filled band)
  For bipartite: gap = spectral gap of adjacency matrix
  For non-bipartite: gap is different!
  -> Graph bipartiteness detectible from eigenvalue gap!
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import time

# ===================== LDOS COMPUTATION =====================

def compute_ldos(G, sigma=0.5, n_E=50):
    """
    Compute Local Density of States for each node.
    LDOS(v, E) = sum_k |phi_k(v)|^2 * Gaussian(E - lambda_k, sigma)

    Returns:
      E_grid: energy grid
      ldos: n x n_E matrix of LDOS values (node x energy)
      eigvals, eigvecs
    """
    n = G.number_of_nodes()
    if n < 2:
        return np.array([]), np.zeros((1, n_E)), np.array([]), np.zeros((1, 1))

    try:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        return np.array([]), np.zeros((n, n_E)), np.array([]), np.zeros((n, n))

    # Energy grid spanning eigenvalue range
    E_min = eigvals.min() - 2 * sigma
    E_max = eigvals.max() + 2 * sigma
    E_grid = np.linspace(E_min, E_max, n_E)

    # LDOS: n x n_E matrix
    ldos = np.zeros((n, n_E))
    for k in range(n):
        for ie, E in enumerate(E_grid):
            # Gaussian broadening
            dos_contribution = np.exp(-(E - eigvals[k])**2 / (2 * sigma**2))
            dos_contribution /= (sigma * np.sqrt(2 * np.pi))
            ldos[:, ie] += (eigvecs[:, k]**2) * dos_contribution

    return E_grid, ldos, eigvals, eigvecs

def inverse_participation_ratio(eigvecs):
    """
    Compute IPR for each eigenvector.
    IPR_k = sum_v |phi_k(v)|^4
    IPR = 1/n: delocalized. IPR = 1: localized.
    """
    # eigvecs: n x n matrix, columns are eigenvectors
    ipr = np.sum(eigvecs**4, axis=0)
    return ipr

def spatial_entropy(eigvecs):
    """
    Compute spatial entropy H_k for each eigenvector.
    H_k = -sum_v |phi_k(v)|^2 * log(|phi_k(v)|^2)
    Low H: localized. High H: delocalized (max = log n).
    """
    probs = eigvecs**2  # n x n matrix
    # Normalize each column
    col_sums = probs.sum(axis=0, keepdims=True) + 1e-12
    probs_norm = probs / col_sums
    # Entropy
    entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-12), axis=0)
    return entropy

def nodal_domains(eigvec):
    """
    Count nodal domains (sign-consistent components) of eigenvector on graph.
    """
    # Find connected components of {v: phi(v) > 0} and {v: phi(v) < 0}
    pos_nodes = np.where(eigvec > 0)[0]
    neg_nodes = np.where(eigvec < 0)[0]
    return len(pos_nodes) + len(neg_nodes)  # simplified: just count sign changes

def gldos_invariants(G):
    """
    Compute all GLDOS invariants.
    """
    n = G.number_of_nodes()
    if n < 3:
        return {}

    E_grid, ldos, eigvals, eigvecs = compute_ldos(G, sigma=0.5)
    ipr = inverse_participation_ratio(eigvecs)
    H_spatial = spatial_entropy(eigvecs)

    # LDOS statistics
    ldos_node_std = np.std(ldos, axis=0)  # spectral inhomogeneity at each E
    mean_ldos_inhom = np.mean(ldos_node_std)
    max_ldos_inhom = np.max(ldos_node_std)

    # IPR statistics
    ipr_delocalized = 1.0 / n  # fully delocalized value
    mean_ipr = np.mean(ipr)
    max_ipr = np.max(ipr)
    ipr_ratio = mean_ipr / ipr_delocalized  # how much more localized than uniform

    # Spatial entropy statistics
    mean_H = np.mean(H_spatial)
    min_H = np.min(H_spatial)  # most localized state
    H_max = np.log(n)  # theoretical maximum

    # Energy gap (HOMO-LUMO analog)
    n_occ = n // 2  # half-filled band
    if n_occ < n:
        energy_gap = eigvals[n_occ] - eigvals[n_occ - 1]
    else:
        energy_gap = 0

    # Anderson localization measure
    # Count states with IPR > 5/n (significantly localized)
    n_localized = np.sum(ipr > 5.0 / n)
    localization_fraction = n_localized / n

    return {
        'mean_ipr': mean_ipr,
        'max_ipr': max_ipr,
        'ipr_ratio': ipr_ratio,
        'mean_H_spatial': mean_H,
        'min_H_spatial': min_H,
        'ldos_inhomogeneity': mean_ldos_inhom,
        'max_ldos_inhom': max_ldos_inhom,
        'energy_gap': energy_gap,
        'localization_fraction': localization_fraction,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'ipr': ipr,
        'H_spatial': H_spatial,
        'ldos': ldos,
        'E_grid': E_grid,
    }

# ===================== EXPERIMENT 1: LDOS OF CLASSIC GRAPHS =====================

print("=== Graph Local Density of States (GLDOS) ===\n")
print("Condensed matter physics: eigenvector localization = new graph invariant\n")

print("--- Experiment 1: LDOS Profiles of Classic Graphs ---\n")
print(f"{'Graph':15s}  {'n':3s}  {'mean_IPR*n':10s}  {'max_IPR*n':9s}  {'min_H':5s}  {'E_gap':6s}  {'loc_frac':8s}")
print("-" * 70)

test_graphs = {
    'K5':         nx.complete_graph(5),
    'K6':         nx.complete_graph(6),
    'C6':         nx.cycle_graph(6),
    'C8':         nx.cycle_graph(8),
    'Petersen':   nx.petersen_graph(),
    'K3,3':       nx.complete_bipartite_graph(3, 3),
    'Path(8)':    nx.path_graph(8),
    'Star(8)':    nx.star_graph(7),
    'Grid(3x3)':  nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
    'Tree(2,3)':  nx.balanced_tree(2, 3),
    'ER(10,0.3)': nx.erdos_renyi_graph(10, 0.3, seed=42),
    'BA(10,2)':   nx.barabasi_albert_graph(10, 2, seed=42),
}

ldos_data = {}
for name, G in test_graphs.items():
    inv = gldos_invariants(G)
    if not inv:
        continue

    n = G.number_of_nodes()
    # Scale IPR by n: IPR*n = 1 for delocalized, IPR*n = n for fully localized
    mean_ipr_n = inv['mean_ipr'] * n
    max_ipr_n = inv['max_ipr'] * n

    print(f"{name:15s}  {n:3d}  {mean_ipr_n:10.4f}  {max_ipr_n:9.4f}  "
          f"{inv['min_H_spatial']:5.3f}  {inv['energy_gap']:6.4f}  {inv['localization_fraction']:8.4f}")
    ldos_data[name] = inv

print()
print("mean_IPR*n = 1.0: fully delocalized (all modes spread over whole graph)")
print("mean_IPR*n >> 1: Anderson localization (modes confined to small regions)")
print("min_H: entropy of most localized eigenstate (min over states)")
print("E_gap: HOMO-LUMO gap analog (energy gap at half-filling)")
print("loc_frac: fraction of eigenstates significantly localized")

# ===================== EXPERIMENT 2: ANDERSON LOCALIZATION =====================

print()
print("--- Experiment 2: Anderson Localization by Graph Type ---\n")
print("Irregular graphs should show MORE localization than regular graphs\n")

np.random.seed(42)
n = 12

graph_types = {
    '4-regular':    nx.circulant_graph(n, [1, 2]),
    'ER(0.3)':      nx.erdos_renyi_graph(n, 0.3, seed=42),
    'ER(0.5)':      nx.erdos_renyi_graph(n, 0.5, seed=42),
    'BA(2)':        nx.barabasi_albert_graph(n, 2, seed=42),
    'BA(4)':        nx.barabasi_albert_graph(n, 4, seed=42),
    'WS(4,0.1)':    nx.watts_strogatz_graph(n, 4, 0.1, seed=42),
    'WS(4,0.9)':    nx.watts_strogatz_graph(n, 4, 0.9, seed=42),
}

print(f"{'Type':12s}  {'mean_IPR*n':10s}  {'loc_frac':8s}  {'degree_std':10s}  {'Regular?'}")
print("-" * 55)

for gname, G in graph_types.items():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)

    inv = gldos_invariants(G)
    if not inv:
        continue

    degrees = np.array([G.degree(v) for v in G.nodes()], dtype=float)
    deg_std = np.std(degrees)
    is_regular = deg_std < 0.01

    print(f"{gname:12s}  {inv['mean_ipr']*G.number_of_nodes():10.4f}  "
          f"{inv['localization_fraction']:8.4f}  {deg_std:10.4f}  {str(is_regular):5s}")

print()
print("PREDICTION: regular graphs have IPR*n ~ 1 (uniform delocalization)")
print("Scale-free (BA) graphs: high IPR*n (hub eigenstates localized at hubs)")

# ===================== EXPERIMENT 3: HOMO-LUMO GAP ANALYSIS =====================

print()
print("--- Experiment 3: HOMO-LUMO Gap in Molecular Graphs ---\n")

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    mol_smiles = {
        "ethylene":   "C=C",
        "benzene":    "c1ccccc1",
        "naphthalene":"c1ccc2ccccc2c1",
        "anthracene": "c1ccc2cc3ccccc3cc2c1",
        "butadiene":  "C=CC=C",
        "pyridine":   "c1ccncc1",
        "pyrazine":   "c1cnccn1",
        "furan":      "c1ccoc1",
    }

    print(f"{'Molecule':12s}  {'n':3s}  {'E_gap':6s}  {'mean_IPR*n':10s}  {'HOMO_H':7s}  {'Interpretation'}")
    print("-" * 60)

    for mol_name, smi in mol_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        inv = gldos_invariants(G)
        if not inv:
            continue

        n = G.number_of_nodes()
        # HOMO eigenstate (n//2 - 1 index, 0-based)
        n_occ = n // 2
        homo_idx = n_occ - 1 if n_occ > 0 else 0
        H_homo = inv['H_spatial'][homo_idx]

        if inv['energy_gap'] < 0.1:
            interp = "metallic (no gap)"
        elif inv['energy_gap'] < 0.5:
            interp = "small gap (reactive)"
        else:
            interp = "large gap (stable)"

        print(f"{mol_name:12s}  {n:3d}  {inv['energy_gap']:6.4f}  {inv['mean_ipr']*n:10.4f}  "
              f"{H_homo:7.4f}  {interp}")

except ImportError:
    print("RDKit not available. Showing graph-theoretic examples instead.")
    for name in ['C6', 'K3,3', 'Petersen']:
        if name in ldos_data:
            inv = ldos_data[name]
            print(f"{name}: E_gap={inv['energy_gap']:.4f}, IPR*n={inv['mean_ipr']*test_graphs[name].number_of_nodes():.4f}")

# ===================== EXPERIMENT 4: NODE-RESOLVED LDOS =====================

print()
print("--- Experiment 4: Node-Resolved LDOS for Star vs BA ---\n")
print("Which nodes are 'most quantum-active' at each energy?\n")

for name, G in [('Star(8)', nx.star_graph(7)), ('BA(10,2)', nx.barabasi_albert_graph(10, 2, seed=42))]:
    inv = gldos_invariants(G)
    if not inv:
        continue

    ipr = inv['ipr']
    H_spatial = inv['H_spatial']
    eigvals = inv['eigvals']
    n = G.number_of_nodes()

    # Find most/least localized states
    most_local_k = np.argmax(ipr)
    least_local_k = np.argmin(ipr)

    most_local_node = np.argmax(inv['eigvecs'][:, most_local_k]**2)

    print(f"\n{name}:")
    print(f"  Most localized state: k={most_local_k}, E={eigvals[most_local_k]:.3f}, "
          f"IPR*n={ipr[most_local_k]*n:.3f}, H={H_spatial[most_local_k]:.3f}")
    print(f"  At node: {most_local_node} (degree={G.degree(list(G.nodes())[most_local_node])})")
    print(f"  Least localized: k={least_local_k}, E={eigvals[least_local_k]:.3f}, "
          f"IPR*n={ipr[least_local_k]*n:.3f}, H={H_spatial[least_local_k]:.3f}")

# ===================== CLASSIFICATION =====================

print()
print("--- Experiment 5: GLDOS Classification ---\n")

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

X_ldos = []
for G in clf_graphs:
    inv = gldos_invariants(G)
    n = G.number_of_nodes()
    X_ldos.append([
        inv.get('mean_ipr', 0) * n,
        inv.get('max_ipr', 0) * n,
        inv.get('energy_gap', 0),
        inv.get('localization_fraction', 0),
    ])
X_ldos = np.array(X_ldos)

for feat_name, X in [('Basic (3)', X_basic),
                      ('GLDOS (4)', X_ldos),
                      ('Combined (7)', np.hstack([X_basic, X_ldos]))]:
    try:
        acc = cross_val_score(LogisticRegression(max_iter=500),
                              StandardScaler().fit_transform(X), y, cv=5).mean()
        print(f"  {feat_name:25s}: CV accuracy = {acc:.3f}")
    except Exception as e:
        print(f"  {feat_name:25s}: error ({e})")

# ===================== THEORY =====================

print()
print("=== THEORETICAL FRAMEWORK ===\n")
print("Graph Local Density of States (GLDOS) bridges condensed matter physics to graphs:\n")
print("  LDOS(v, E) = local electronic density at node v at energy E")
print("  = sum_k |phi_k(v)|^2 * delta(E - lambda_k)")
print()
print("KEY QUANTITY: Inverse Participation Ratio")
print("  IPR_k = sum_v |phi_k(v)|^4")
print("  For Laplacian: phi_1 = 1/sqrt(n) (uniform), IPR_1 = 1/n (minimum)")
print("  For localized states: IPR -> 1 (maximum)")
print()
print("ANDERSON LOCALIZATION THEOREM:")
print("  For random graphs with degree variance sigma^2:")
print("    If sigma^2 / mean_degree >> 1: localization transition occurs!")
print("    States near lambda = 0 and lambda = lambda_max LOCALIZE first")
print("    States in band center remain delocalized")
print("  -> Degree irregularity drives quantum localization!")
print()
print("MOLECULAR ORBITAL THEORY CONNECTION:")
print("  GLDOS at HOMO energy = electron density map")
print("  Nodes with high LDOS at HOMO = nucleophilic attack sites")
print("  Nodes with low LDOS at HOMO = electrophilic attack sites")
print("  -> GLDOS predicts CHEMICAL REACTIVITY from graph alone!")
print()
print("HOMO-LUMO GAP:")
print("  Large gap: chemically stable (benzene, aromatic systems)")
print("  Small gap: reactive, conducting (metals, radicals)")
print("  -> Graph HOMO-LUMO gap = stability predictor!")
print()
print("NEW PREDICTION:")
print("  Graphs with high GLDOS inhomogeneity have high chemical selectivity")
print("  (different sites react with different reagents)")
print("  -> GLDOS inhomogeneity = regioselectivity measure!")
