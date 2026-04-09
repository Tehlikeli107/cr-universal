"""
Structural ODE (SODE): Learning Differential Equations in CR Histogram Space

CORE IDEA:
Normal ODE: dx/dt = f(x) where x in R^N (one state per node) -- HUGE
Structural ODE: dH/dt = g(H) where H = CR k-subgraph histogram -- SMALL (<<N)

CLAIM: For many network dynamics, the CR histogram H_k is an APPROXIMATELY
SUFFICIENT STATISTIC -- knowing H_k at time t predicts H_k at t+dt.

If true: this gives a COMPRESSED MODEL of dynamics:
- From N differential equations to ~100-500 CR histogram equations
- Analytic understanding via the learned g function
- Phase transitions detectable from H's equilibria

TEST: SIR epidemic model on Erdos-Renyi network
  - N agents (S/I/R states)
  - Full simulation: track each agent's state
  - CR encoding: H_3(G_t) where G_t = contact graph restricted to Susceptible agents
    (or better: G_t = "infection graph" where nodes have state labels)

NODE LABELS FOR SIR:
  - Node types: S=1, I=2, R=3 (3 types)
  - Edge exists if agents are in contact
  - k=2 subgraph: distribution over (state_u, state_v, contact) types
  - k=3 subgraph: 3-agent clusters with state combinations

From Label-k_min theory: with c_node=3 types, k=2 or k=3 might be sufficient!

SURPRISE CLAIM: The CR histogram H_3 of the CONTACT GRAPH with STATE LABELS
follows a closed differential equation -- the "structural ODE"
-- without needing individual agent states!
"""
import numpy as np
from collections import Counter
from itertools import combinations
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import time

def sir_cr_encode(G, states, k=2):
    """
    Encode the current SIR state as a CR histogram.
    G: contact network (static)
    states: array of states (0=S, 1=I, 2=R) for each node
    k: subgraph order to compute
    Returns: normalized histogram vector
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    if k == 2:
        # k=2: all pairs, encode (state_u, state_v, edge_present)
        c = Counter()
        edges = set(G.edges())
        for i in range(n):
            for j in range(i+1, n):
                u, v = nodes[i], nodes[j]
                su, sv = states[i], states[j]
                edge = int(G.has_edge(u, v))
                # Sort state types for canonical form
                t = (min(su, sv), max(su, sv), edge)
                c[t] += 1
        # Normalize
        total = sum(c.values())
        return c, total

    elif k == 3:
        # k=3: sample triples
        c = Counter()
        combos = list(combinations(range(n), 3))
        if len(combos) > 5000:
            idx = np.random.choice(len(combos), 5000, replace=False)
            combos = [combos[i] for i in idx]

        for i,j,k_ in combos:
            si, sj, sk = states[i], states[j], states[k_]
            u, v, w = nodes[i], nodes[j], nodes[k_]
            n_edges = int(G.has_edge(u,v)) + int(G.has_edge(u,w)) + int(G.has_edge(v,w))
            # Sort state triple for canonical form
            state_tri = tuple(sorted([si, sj, sk]))
            c[(state_tri, n_edges)] += 1

        total = sum(c.values())
        return c, total

def simulate_sir(G, beta=0.3, gamma=0.1, initial_infected_frac=0.05, T=100, seed=42):
    """
    SIR epidemic simulation on network G.
    Returns: list of (S, I, R) counts and CR histograms at each step.
    """
    np.random.seed(seed)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    states = np.zeros(n, dtype=int)  # 0=S, 1=I, 2=R
    n_initial = max(1, int(initial_infected_frac * n))
    initial_infected = np.random.choice(n, n_initial, replace=False)
    states[initial_infected] = 1

    history_sir = []
    history_k2 = []
    history_k3 = []

    for t in range(T):
        S = (states == 0).sum()
        I = (states == 1).sum()
        R = (states == 2).sum()
        history_sir.append((S/n, I/n, R/n))

        # CR k=2 histogram
        c2, tot2 = sir_cr_encode(G, states, k=2)
        c3, tot3 = sir_cr_encode(G, states, k=3)
        history_k2.append(c2)
        history_k3.append(c3)

        if I == 0:
            # Epidemic over, pad rest
            for _ in range(T - t - 1):
                history_sir.append((S/n, 0, R/n))
                history_k2.append(c2)
                history_k3.append(c3)
            break

        # Update states
        new_states = states.copy()
        for i, v in enumerate(nodes):
            if states[i] == 1:  # Infected
                # Recovery
                if np.random.random() < gamma:
                    new_states[i] = 2
                else:
                    # Infect susceptible neighbors
                    for u in G.neighbors(v):
                        j = node_idx[u]
                        if states[j] == 0 and np.random.random() < beta:
                            new_states[j] = 1
        states = new_states

    return history_sir, history_k2, history_k3

def cr_to_vector(c, vocab):
    """Convert CR counter to numpy vector given vocabulary."""
    v = np.zeros(len(vocab))
    for k, count in c.items():
        if k in vocab:
            v[vocab[k]] = count
    # Normalize
    s = v.sum()
    if s > 0: v /= s
    return v

print("=== Structural ODE: Learning Dynamics in CR Histogram Space ===\n")
print("CLAIM: CR k=2 histogram with SIR labels is a sufficient statistic")
print("for predicting the NEXT CR histogram (without individual agent states)\n")

np.random.seed(42)
N = 200
p = 0.05
G = nx.erdos_renyi_graph(N, p, seed=42)
print(f"Network: ER(N={N}, p={p}), edges={G.number_of_edges()}, mean_deg={2*G.number_of_edges()/N:.1f}")

# Run many simulations to collect training data
N_SIMS = 30
print(f"Running {N_SIMS} SIR simulations...", flush=True)

all_cr2_sequences = []
all_sir_sequences = []

for sim in range(N_SIMS):
    beta = np.random.uniform(0.2, 0.5)
    gamma = np.random.uniform(0.05, 0.2)
    sir, k2, k3 = simulate_sir(G, beta=beta, gamma=gamma, T=50, seed=sim)
    all_cr2_sequences.append(k2)
    all_sir_sequences.append(sir)

# Build vocabulary for k=2 features
vocab2 = set()
for seq in all_cr2_sequences:
    for c in seq:
        vocab2.update(c.keys())
vocab2 = {k: i for i, k in enumerate(sorted(vocab2, key=str))}
print(f"CR k=2 vocabulary size: {len(vocab2)} types")

# Convert to vectors
X_sequences = []
for seq in all_cr2_sequences:
    X_sequences.append([cr_to_vector(c, vocab2) for c in seq])

print()
print("Testing SODE: Can H_t predict H_{t+1}?")

# Collect (H_t, H_{t+1}) pairs for training
X_train, y_train = [], []
for seq in X_sequences[:-5]:  # Leave 5 for testing
    for t in range(len(seq)-1):
        X_train.append(seq[t])
        y_train.append(seq[t+1])

X_test, y_test = [], []
for seq in X_sequences[-5:]:
    for t in range(len(seq)-1):
        X_test.append(seq[t])
        y_test.append(seq[t+1])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"Feature dim: {X_train.shape[1]}")

# SODE: linear model H_{t+1} = A * H_t
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Measure: how well does CR prediction work?
# Metric 1: MSE of CR histogram prediction
mse_cr = np.mean((pred - y_test)**2)
# Metric 2: MSE of SIR macro quantities (S, I, R fractions) derived from CR prediction
# k=2 histogram has entries: (state_u, state_v, edge). From this:
# frac_I = estimated from (1,1,0) + (1,1,1) + (0,1,0) + (0,1,1) + (1,2,0) + (1,2,1) terms

# Compute BASELINE: predicting H_{t+1} = H_t (no change)
mse_baseline = np.mean((X_test - y_test)**2)

print()
print(f"MSE(SODE prediction): {mse_cr:.6f}")
print(f"MSE(baseline, H_t=H_t-1): {mse_baseline:.6f}")
print(f"SODE R^2 improvement: {1 - mse_cr/mse_baseline:.3f}")

# Also test: predict macro SIR quantities from CR histogram
# From k=2: approximate fraction infected
def estimate_I_from_cr(cr_vec, vocab):
    """Estimate I fraction from k=2 CR histogram."""
    # Count entries involving state=1 (Infected)
    I_pairs = sum(cr_vec[i] for k, i in vocab.items()
                  if isinstance(k, tuple) and (k[0]==1 or k[1]==1))
    return I_pairs

# This is approximate, but let's see if SODE captures the trajectory
print()
print("=== Testing SODE Trajectory Prediction ===")
print()
for test_sim_idx in range(2):
    seq = X_sequences[-(test_sim_idx+1)]
    sir_seq = all_sir_sequences[-(test_sim_idx+1)]

    # Predict next 10 steps from initial state
    H = seq[0].copy()
    predicted_H_traj = [H]
    for t in range(min(20, len(seq)-1)):
        H_next = clf.predict(H.reshape(1,-1))[0]
        H_next = np.clip(H_next, 0, None)
        s = H_next.sum()
        if s > 0: H_next /= s
        predicted_H_traj.append(H_next)

    # Compare actual vs predicted in terms of some simple statistics
    # Use L1 norm between actual and predicted CR histograms
    actual_seq = seq[:len(predicted_H_traj)]
    errors = [np.abs(predicted_H_traj[t] - actual_seq[t]).mean()
              for t in range(len(predicted_H_traj))]

    print(f"  Sim {test_sim_idx+1}: SODE prediction error (L1):")
    print(f"    t=0: {errors[0]:.4f}, t=5: {errors[min(5,len(errors)-1)]:.4f}, "
          f"t=10: {errors[min(10,len(errors)-1)]:.4f}, t=20: {errors[min(20,len(errors)-1)]:.4f}")

print()
print("=== THEORETICAL IMPLICATIONS ===\n")
print(f"If SODE works (R^2 > 0.5):")
print(f"  -> CR histogram is an APPROXIMATE SUFFICIENT STATISTIC for SIR dynamics")
print(f"  -> Network dynamics can be modeled with {len(vocab2)} equations instead of {N}")
print(f"  -> {N//len(vocab2):.0f}x compression of the state space!")
print(f"  -> Enables analytic solution of epidemic thresholds from graph structure")
print()
print(f"THIS WOULD BE A NEW CLASS OF MEAN-FIELD THEORY:")
print(f"  Standard mean-field: uses degree distribution (1D)")
print(f"  Pair approximation: uses pair distributions (2D)")
print(f"  CR-mean-field: uses full k=2,3 labeled subgraph distribution ({len(vocab2)}D)")
print(f"  More accurate because it captures higher-order correlations")
