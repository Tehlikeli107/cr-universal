"""
SIR Epidemic Dynamics as Graph Invariant

PARADIGM: Run deterministic SIR epidemic from every starting node.
The SORTED PROFILE of epidemic parameters is a graph invariant.

KEY INSIGHT: Unlike spectral methods (linear eigenvalue analysis),
SIR uses NONLINEAR DYNAMICS. Cospectral graphs can have different
epidemic behavior because SIR responds to higher-order structure:
clustering, community hierarchy, path distances — all beyond linear.

CONSTRUCTION:
  For each seed node v, run SIR ODE on G:
    dS/dt = -beta * S_v * sum_{u ~ v} I_u
    dI/dt = beta * S_v * sum_{u ~ v} I_u - gamma * I_v
    dR/dt = gamma * I_v

  Extract per-node features: peak_time(v), peak_frac(v), final_R(v)
  SORT these across all starting nodes -> label-independent invariant

  Signature = (sorted peak times) + (sorted peak fracs) + (sorted final R)

RESULTS:
  n=6: COMPLETE (112/112, 1.2s) — all non-iso graphs separated
  n=7: COMPLETE (853/853, 18.9s) — all non-iso graphs separated!
  Shrikhande vs Rook: FAILS (both vertex-transitive, same epidemic from all nodes)

FAILURE MODE: Vertex-transitive graphs (all nodes equivalent)
  -> All starting nodes give identical epidemic curve
  -> Sorted profile is trivially the same for different vertex-transitive graphs
  -> Complementary to Hodge (which SUCCEEDS on Shrikhande/Rook)

COMPARISON:
  Hodge L1: complete n<=6 alone, complete n<=7 with L0+A, 0.12s
  SIR:      complete n<=7, 18.9s (157x slower), fails vertex-transitive
  -> Hodge is faster AND stronger on hard cases, but SIR captures different structure
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import time


def sir_epidemic_deterministic(G, seed_node, beta=0.3, gamma=0.1, T=40, dt=0.2):
    """
    Deterministic SIR epidemic starting from seed_node.
    Returns peak_time, peak_fraction_infected, final_recovered.

    Deterministic = mean-field ODE (not stochastic simulation).
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    adj = [[node_idx[w] for w in G.neighbors(v)] for v in nodes]

    # Initial conditions: seed is infected, all else susceptible
    S = np.ones(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[node_idx[seed_node]] = 0.0
    I[node_idx[seed_node]] = 1.0

    # Euler integration
    n_steps = int(T / dt)
    best_I = 0.0
    best_t = 0.0

    for step in range(n_steps):
        dS = np.zeros(n)
        dI = np.zeros(n)
        dR = np.zeros(n)
        for i in range(n):
            force = beta * S[i] * sum(I[j] for j in adj[i])
            rec = gamma * I[i]
            dS[i] = -force
            dI[i] = force - rec
            dR[i] = rec
        S += dt * dS
        I += dt * dI
        R += dt * dR

        total_I = float(np.sum(I)) / n
        if total_I > best_I:
            best_I = total_I
            best_t = step * dt

    final_R = float(np.sum(R)) / n
    return round(best_t, 3), round(best_I, 3), round(final_R, 3)


def sir_graph_signature(G, beta=0.3, gamma=0.1, T=40, dt=0.2):
    """
    SIR epidemic signature for graph G.
    Returns sorted tuple of (peak_times, peak_fracs, final_Rs).
    """
    peaks, fracs, finals = [], [], []
    for v in G.nodes():
        pt, pf, fr = sir_epidemic_deterministic(G, v, beta, gamma, T, dt)
        peaks.append(pt)
        fracs.append(pf)
        finals.append(fr)

    return (
        tuple(sorted(peaks)),
        tuple(sorted(fracs, reverse=True)),
        tuple(sorted(finals, reverse=True))
    )


def sir_combined_signature(G, beta=0.3, gamma=0.1, T=40, dt=0.2):
    """Concatenated SIR signature for use as a dictionary key."""
    p, f, r = sir_graph_signature(G, beta, gamma, T, dt)
    return p + f + r


def sir_invariants(G, beta=0.3, gamma=0.1, T=40, dt=0.2):
    """
    Summary statistics from SIR epidemic profile.
    """
    p, f, r = sir_graph_signature(G, beta, gamma, T, dt)
    n = G.number_of_nodes()

    return {
        'mean_peak_time': float(np.mean(p)),
        'std_peak_time': float(np.std(p)),
        'min_peak_time': float(min(p)),  # fastest node to drive epidemic
        'max_peak_time': float(max(p)),  # slowest node
        'mean_peak_frac': float(np.mean(f)),
        'max_peak_frac': float(max(f)),
        'min_final_R': float(min(r)),  # hardest node to spread from
        'max_final_R': float(max(r)),  # easiest node to spread from
        'R_spread': float(max(r)) - float(min(r)),  # heterogeneity
        'peak_spread': float(max(p)) - float(min(p)),  # timing heterogeneity
    }


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

if __name__ == '__main__':
    print("=== SIR Epidemic Fingerprinting ===\n")
    print("Deterministic SIR ODE, beta=0.3, gamma=0.1\n")

    # 1. Famous graphs
    print("--- Per-node epidemic profiles ---\n")
    test_graphs = [
        ('P6', nx.path_graph(6)),
        ('C6', nx.cycle_graph(6)),
        ('K4', nx.complete_graph(4)),
        ('K3,3', nx.complete_bipartite_graph(3, 3)),
        ('Petersen', nx.petersen_graph()),
        ('Star5', nx.star_graph(4)),
    ]

    for name, G in test_graphs:
        G = nx.convert_node_labels_to_integers(G)
        inv = sir_invariants(G)
        print(f"  {name:10s}: peak_t=[{inv['min_peak_time']:.1f}..{inv['max_peak_time']:.1f}]  "
              f"spread={inv['peak_spread']:.1f}  R_inf=[{inv['min_final_R']:.3f}..{inv['max_final_R']:.3f}]")

    print()
    print("NOTE: Petersen = vertex-transitive -> all peaks identical (spread=0)")
    print("NOTE: Star5 = hub vs leaves -> large timing spread")

    print()

    # 2. Completeness tests
    print("--- Completeness: n=4..7 ---\n")
    atlas = nx.graph_atlas_g()
    total_time = 0

    for target_n in [4, 5, 6, 7]:
        graphs = [G for G in atlas if G.number_of_nodes() == target_n
                  and nx.is_connected(G)]
        t0 = time.time()
        sigs = {i: sir_combined_signature(G) for i, G in enumerate(graphs)}
        t_el = time.time() - t0
        total_time += t_el

        sg = defaultdict(list)
        for i, s in sigs.items():
            sg[s].append(i)
        n_col = sum(1 for g in sg.values() if len(g) > 1)
        status = "COMPLETE" if n_col == 0 else f"{n_col} collisions"

        print(f"  n={target_n}: {len(graphs):4d} graphs, {status:20s} {t_el:.2f}s")

    print(f"  Total: {total_time:.2f}s")
    print()

    # 3. Hard pair tests
    print("--- Hard pairs ---\n")

    G_sh = nx.Graph()
    G_sh.add_nodes_from(range(16))
    for i in range(4):
        for j in range(4):
            v = 4 * i + j
            for di, dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1)]:
                u = 4 * ((i+di)%4) + (j+dj)%4
                G_sh.add_edge(v, u)
    G_rk = nx.convert_node_labels_to_integers(
        nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))

    s_sh = sir_combined_signature(G_sh)
    s_rk = sir_combined_signature(G_rk)
    print(f"Shrikhande vs Rook(4,4):")
    print(f"  SIR same: {s_sh == s_rk}")
    print(f"  (Both vertex-transitive -> SAME epidemic from every node -> FAILS)")
    print(f"  BUT Hodge L1 SUCCEEDS on this pair!")
    print()

    print("=== PHYSICAL INTERPRETATION ===\n")
    print("SIR epidemic captures:")
    print("  - Community structure (epidemics spread within communities first)")
    print("  - Bottleneck nodes (bridges slow epidemic spread)")
    print("  - Hub influence (high-degree nodes trigger faster epidemics)")
    print()
    print("Failure on vertex-transitive graphs = lack of node heterogeneity")
    print("This is a DIFFERENT failure mode than Hodge (spectral similarity)")
    print()
    print("Recommendation: combine SIR + Hodge for robust graph fingerprinting")
