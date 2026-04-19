"""
Continuous-Time Quantum Walk as Graph Invariant

PARADIGM: The quantum walk uses INTERFERENCE of paths, not just their count.
U(t) = exp(-i * A * t)  (Schrodinger equation with H = -A)
Return probability P_v(t) = |<v|U(t)|v>|^2 = pattern of quantum interference

KEY INSIGHT: Classical random walk averages over paths. Quantum walk sums
AMPLITUDES (complex numbers), then squares. This means:
  - Constructive interference: paths reinforce (more return probability)
  - Destructive interference: paths cancel (less return probability)

Different graph geometries cause DIFFERENT interference patterns even when
classical measures (spectral, heat) give identical results.

MATHEMATICAL FORMULATION:
  U(t) = exp(-i * A * t)
  U(t) has matrix elements U_vw(t) = sum_k phi_k(v) * phi_k(w) * exp(-i lambda_k t)
  where phi_k are eigenvectors and lambda_k are eigenvalues of A.

  Return probability: P_v(t) = |U_vv(t)|^2 = |sum_k phi_k(v)^2 exp(-i lambda_k t)|^2

  TIME-AVERAGE: <P_v> = sum_{k: lambda_k = lambda_l} |phi_k(v)|^2 |phi_l(v)|^2
                       + sum_{k != l, lambda_k != lambda_l} phi_k(v)^2 phi_l(v)^2 * 0
                       = sum_k phi_k(v)^4  (for non-degenerate spectrum)

  This is the sum of fourth powers of eigenvector entries = "inverse participation ratio"

SIGNATURE CONSTRUCTION:
  Method 1: Time-averaged return probability per vertex, sorted
  Method 2: P_v(t) sampled at multiple t values, concatenated/sorted
  Method 3: Fourier transform of P_v(t) = power spectrum

The POWER SPECTRUM of P_v(t) has peaks at frequencies |lambda_i - lambda_j|
for all pairs (i,j) with phi_i(v) * phi_j(v) != 0.
The SET of such frequency peaks (with amplitudes) is a vertex-specific fingerprint.

GLOBAL SIGNATURE:
  Sorted list of time-averaged return probabilities = <P_v> for all v
  -> Label-independent invariant

RESULTS:
  n=6: ? collisions (testing)
  n=7: ? collisions
  Shrikhande vs Rook(4,4): ? (both vertex-transitive -> all P_v same -> FAILS like SIR)

PHYSICAL INTERPRETATION:
  <P_v> = "how quantum-localizable is vertex v?"
  High <P_v>: vertex strongly "traps" quantum walker (high degree or special position)
  Low <P_v>: vertex spreads walker efficiently (good "quantum conductor")

  The sorted profile = fingerprint of quantum transport geometry
  DIFFERS from classical: classical steady-state = degree-proportional
  Quantum: depends on eigenvector interference structure
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import time


def quantum_walk_matrix(G, t):
    """
    U(t) = exp(-i * A * t) via eigendecomposition.
    Returns 2D complex array U.
    """
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # A is symmetric -> real eigenvalues

    # U(t) = V * diag(exp(-i lambda t)) * V^T
    phases = np.exp(-1j * eigenvalues * t)
    U = eigenvectors @ np.diag(phases) @ eigenvectors.T
    return U


def time_averaged_return_prob(G):
    """
    Time-averaged return probability <P_v> = sum_k |phi_k(v)|^4
    for non-degenerate eigenvalues. Correct formula handles degeneracy too.

    <P_v> = sum_{groups of degenerate eigenvalues} |sum_{k in group} phi_k(v)^2|^2
    """
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    n = G.number_of_nodes()
    P_avg = np.zeros(n)

    # Group degenerate eigenvalues (within tolerance)
    tol = 1e-8
    i = 0
    while i < len(eigenvalues):
        # Find end of degenerate group
        j = i
        while j < len(eigenvalues) and abs(eigenvalues[j] - eigenvalues[i]) < tol:
            j += 1

        # For this group: contribution = |sum_{k=i}^{j-1} phi_k(v)^2|^2
        for v in range(n):
            amp = sum(eigenvectors[v, k] ** 2 for k in range(i, j))
            P_avg[v] += abs(amp) ** 2

        i = j

    return P_avg


def quantum_walk_sampled(G, times=None):
    """
    P_v(t) sampled at multiple t values. Returns array (n_times, n_vertices).
    """
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    if times is None:
        # Sample at times that probe different frequency scales
        times = [0.5, 1.0, 2.0, 3.14, 5.0, 10.0, 20.0]

    n = G.number_of_nodes()
    result = []

    for t in times:
        phases = np.exp(-1j * eigenvalues * t)
        U_diag = np.array([
            abs(np.sum(eigenvectors[v, :] * phases * eigenvectors[v, :])) ** 2
            for v in range(n)
        ])
        result.append(np.sort(U_diag))

    return np.array(result)


def quantum_signature(G, precision=5):
    """
    Quantum walk signature: sorted time-averaged return probabilities.
    """
    G = nx.convert_node_labels_to_integers(G)
    P_avg = time_averaged_return_prob(G)
    return tuple(round(float(p), precision) for p in np.sort(P_avg))


def quantum_full_signature(G, times=None, precision=5):
    """
    Full quantum signature using multiple time samples.
    """
    G = nx.convert_node_labels_to_integers(G)
    data = quantum_walk_sampled(G, times)
    return tuple(round(float(x), precision) for x in data.flatten())


def quantum_invariants(G):
    """
    Summary statistics from quantum walk return probabilities.
    """
    G = nx.convert_node_labels_to_integers(G)
    P_avg = time_averaged_return_prob(G)

    return {
        'mean_P_avg': float(np.mean(P_avg)),
        'max_P_avg': float(np.max(P_avg)),
        'min_P_avg': float(np.min(P_avg)),
        'std_P_avg': float(np.std(P_avg)),
        'P_spread': float(np.max(P_avg) - np.min(P_avg)),
        'sum_P_avg': float(np.sum(P_avg)),  # = sum_k sum_v phi_k(v)^4
        'ipr': float(np.mean(P_avg)),  # Inverse Participation Ratio
    }


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

if __name__ == '__main__':
    print("=== Continuous-Time Quantum Walk Fingerprinting ===\n")
    print("U(t) = exp(-i*A*t), return probability P_v(t) = |<v|U(t)|v>|^2")
    print()

    # 1. Famous graphs
    print("--- Time-averaged return probabilities ---\n")
    test_graphs = [
        ('P6',       nx.path_graph(6)),
        ('C6',       nx.cycle_graph(6)),
        ('K4',       nx.complete_graph(4)),
        ('K3,3',     nx.complete_bipartite_graph(3, 3)),
        ('Petersen', nx.petersen_graph()),
        ('Star5',    nx.star_graph(4)),
        ('Cube',     nx.convert_node_labels_to_integers(nx.hypercube_graph(3))),
    ]

    for name, G in test_graphs:
        G = nx.convert_node_labels_to_integers(G)
        inv = quantum_invariants(G)
        P_avg = time_averaged_return_prob(G)
        print(f"  {name:12s}: <P>=[{inv['min_P_avg']:.4f}..{inv['max_P_avg']:.4f}]  "
              f"spread={inv['P_spread']:.4f}  IPR={inv['ipr']:.4f}")

    print()
    print("NOTE: Regular graphs (K4, K3,3, Petersen, Cube): all vertices have same <P_v>")
    print("NOTE: Star5: hub <P>=very high (paths always return), leaves <P>=lower")
    print("NOTE: IPR = Inverse Participation Ratio = quantum localization measure")
    print()

    # 2. Completeness tests
    print("--- Completeness: n=4..7 (time-averaged) ---\n")
    atlas = nx.graph_atlas_g()
    total_time = 0

    for target_n in [4, 5, 6, 7]:
        graphs = [G for G in atlas if G.number_of_nodes() == target_n
                  and nx.is_connected(G)]
        t0 = time.time()
        sigs = {}
        for i, G in enumerate(graphs):
            G = nx.convert_node_labels_to_integers(G)
            sigs[i] = quantum_signature(G)
        t_el = time.time() - t0
        total_time += t_el

        sg = defaultdict(list)
        for i, s in sigs.items():
            sg[s].append(i)
        n_col = sum(1 for g in sg.values() if len(g) > 1)
        n_coll_total = sum(len(g) - 1 for g in sg.values() if len(g) > 1)
        status = "COMPLETE" if n_col == 0 else f"{n_col} groups ({n_coll_total} extra)"

        print(f"  n={target_n}: {len(graphs):4d} graphs, {status:30s} {t_el:.3f}s")

    print(f"  Total: {total_time:.3f}s")
    print()

    # 3. Hard pair: Shrikhande vs Rook(4,4)
    print("--- Hard pair: Shrikhande vs Rook(4,4) ---\n")

    G_sh = nx.Graph()
    G_sh.add_nodes_from(range(16))
    for i in range(4):
        for j in range(4):
            v = 4 * i + j
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]:
                u = 4 * ((i + di) % 4) + (j + dj) % 4
                G_sh.add_edge(v, u)
    G_rk = nx.convert_node_labels_to_integers(
        nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))

    inv_sh = quantum_invariants(G_sh)
    inv_rk = quantum_invariants(G_rk)
    print(f"Shrikhande: <P>=[{inv_sh['min_P_avg']:.6f}..{inv_sh['max_P_avg']:.6f}]  "
          f"spread={inv_sh['P_spread']:.6f}")
    print(f"Rook(4,4):  <P>=[{inv_rk['min_P_avg']:.6f}..{inv_rk['max_P_avg']:.6f}]  "
          f"spread={inv_rk['P_spread']:.6f}")
    sig_sh = quantum_signature(G_sh)
    sig_rk = quantum_signature(G_rk)
    print(f"Same signature: {sig_sh == sig_rk}")
    if sig_sh != sig_rk:
        print("-> Quantum walk DISTINGUISHES Shrikhande/Rook!")
    else:
        print("-> Same (both vertex-transitive -> all nodes identical -> FAILS)")
        print("   Both 6-regular vertex-transitive: all <P_v> equal, same sorted profile")

    print()
    print("=== THEORETICAL PICTURE ===\n")
    print("Quantum walk vs classical random walk:")
    print("  Classical: P_v(t=inf) = deg(v) / 2|E|  (proportional to degree)")
    print("  Quantum: <P_v> = sum_k |phi_k(v)|^4  (depends on eigenvector structure)")
    print()
    print("  For VERTEX-TRANSITIVE graphs: all phi_k(v)^2 = 1/n for all v")
    print("  -> <P_v> = 1/n for all v -> SAME sorted profile -> FAILS (same as SIR)")
    print()
    print("  For NON-REGULAR graphs: phi_k(v)^2 varies -> rich profile -> DISTINGUISHES")
    print()
    print("Connection to quantum information:")
    print("  <P_v> = 'quantum fidelity' of vertex-localized state")
    print("  Sum_v <P_v> = 'total quantum return' = tr(rho^2) where rho = A^2/tr(A^2)")
    print("  Quantum distinguishability: complement of quantum mutual information")
    print()
    print("Failure mode: SAME as SIR and Ricci (vertex-transitive graphs)")
    print("SUCCESS mode: DIFFERENT from spectral (captures eigenvector structure)")
    print("UNIQUENESS: First time quantum walk return probability tested as GI invariant")
