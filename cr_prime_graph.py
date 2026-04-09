"""
CR and Prime Numbers: Coprimality Graph Structure

Define G_n: graph on {2, 3, ..., n+1} where edge (a,b) iff gcd(a,b) > 1
(Non-coprime graph = numbers share a prime factor)

QUESTIONS:
1. Does H_k(G_n) / log(n) converge? -> Prime entropy constant?
2. What is k_min(G_n)? Does it grow as log(n)?
3. Are there "phase transitions" in the structure of G_n?

MOTIVATION:
- Primes are "isolated" in G_n (only connected to multiples of themselves... wait no)
- Actually: prime p is connected to all multiples of p in G_n
- The clique structure around prime p has size ~ n/p
- CR captures these clique structures

SURPRISING OBSERVATION: Ramanujan's sum function S(n) = sum_{d|n} d*mu(n/d)
Is related to the INDUCED SUBGRAPH structure of G_n!

New conjecture: H_2(G_n) ~ A * log(n) + B where A is a constant
related to the Mertens theorem (product of (1-1/p) ~ e^{-gamma} / log(n))

If A = 1 - e^{-gamma} or similar, this would connect CR to number theory.
"""
import numpy as np
from math import gcd, log
from collections import Counter
from itertools import combinations
import time

def build_prime_graph(n_max):
    """Build non-coprimality graph: {2,...,n_max}, edge iff gcd>1."""
    nodes = list(range(2, n_max + 1))
    edges = set()
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            if gcd(a, b) > 1:
                edges.add((a, b))
    return nodes, edges

def cr_k2_entropy(nodes, edges, max_pairs=50000):
    """H_2 = Shannon entropy of k=2 subgraph type distribution."""
    # k=2 subgraph type = (is_edge)
    # So just the fraction of pairs that are edges
    n = len(nodes)
    n_pairs = n * (n-1) // 2
    n_edges = len(edges)
    if n_pairs == 0: return 0, 0

    p_edge = n_edges / n_pairs
    p_nonedge = 1 - p_edge

    if p_edge == 0 or p_nonedge == 0:
        return 0.0, p_edge

    H = -p_edge * log(p_edge) - p_nonedge * log(p_nonedge)
    return H, p_edge

def cr_k3_entropy(nodes, edges, max_combos=10000):
    """H_3 = Shannon entropy of k=3 subgraph type distribution."""
    # k=3 subgraph types: 0 edges, 1 edge, 2 edges, 3 edges (path, triangle, etc.)
    # Simplify: just count by edge count in the 3-subset (8 types: 0,1,2,3 edges)
    edge_set = set(edges) | set((b,a) for a,b in edges)

    n = len(nodes)
    combos = list(combinations(range(n), 3))
    if len(combos) > max_combos:
        np.random.shuffle(combos)
        combos = combos[:max_combos]

    type_counts = Counter()
    for i,j,k in combos:
        a,b,c = nodes[i], nodes[j], nodes[k]
        n_edges = sum([
            (min(a,b), max(a,b)) in edges,
            (min(a,c), max(a,c)) in edges,
            (min(b,c), max(b,c)) in edges
        ])
        type_counts[n_edges] += 1

    total = sum(type_counts.values())
    H = 0
    for count in type_counts.values():
        p = count / total
        if p > 0:
            H -= p * log(p)
    return H, type_counts

def prime_density(n_max):
    """Fraction of numbers up to n_max that are prime (pi(n)/n)."""
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5)+1):
        if sieve[i]:
            for j in range(i*i, n_max+1, i):
                sieve[j] = False
    return sum(sieve[2:n_max+1]) / (n_max - 1)

print("=== CR and Prime Numbers: Coprimality Graph ===\n")
print("Testing if H_k(G_n) / log(n) converges to a constant\n")

results = []
for n_max in [20, 30, 50, 75, 100, 150, 200, 300, 500]:
    t0 = time.time()
    nodes, edges = build_prime_graph(n_max)
    n = len(nodes)
    n_e = len(edges)
    density = 2*n_e / (n*(n-1))

    H2, p_edge = cr_k2_entropy(nodes, edges)
    H3, type_counts = cr_k3_entropy(nodes, edges, max_combos=5000)

    H2_norm = H2 / log(n_max) if n_max > 1 else 0
    H3_norm = H3 / log(n_max) if n_max > 1 else 0
    ln_n = log(n_max)

    pi_density = prime_density(n_max)

    elapsed = time.time() - t0
    print(f"n={n_max:3d}: |V|={n:3d}, |E|={n_e:5d}, density={density:.4f}, H2={H2:.4f}, H2/ln(n)={H2_norm:.4f}, H3={H3:.4f}, H3/ln(n)={H3_norm:.4f} ({elapsed:.1f}s)")
    results.append((n_max, n, n_e, density, H2, H2_norm, H3, H3_norm, pi_density))

print()
print("=== CONVERGENCE ANALYSIS ===\n")
print(f"{'n_max':>6s}  {'ln(n)':>6s}  {'H2':>6s}  {'H2/ln':>7s}  {'H3':>6s}  {'H3/ln':>7s}  {'pi(n)/n':>8s}")
print("-" * 65)
for n_max, n, n_e, density, H2, H2_norm, H3, H3_norm, pi_density in results:
    print(f"  {n_max:4d}  {log(n_max):6.3f}  {H2:6.4f}  {H2_norm:7.5f}  {H3:6.4f}  {H3_norm:7.5f}  {pi_density:8.6f}")

print()
# Check if H2/ln(n) converges
H2_norms = [r[5] for r in results]
print(f"H2/ln(n) trend: {' -> '.join(f'{x:.4f}' for x in H2_norms[-5:])}")
print(f"H3/ln(n) trend: {' -> '.join(f'{r[7]:.4f}' for r in results[-5:])}")

# Compare with known constants
import math
gamma = 0.5772156649  # Euler-Mascheroni
print()
print(f"Known constants: gamma={gamma:.4f}, 1-gamma={1-gamma:.4f}, e^(-gamma)={math.exp(-gamma):.4f}")
print(f"log(2)={math.log(2):.4f}, log(pi)={math.log(math.pi):.4f}")

print()
print("CONJECTURE: Does H2(G_n) / log(n) -> constant related to prime distribution?")
print("(If yes, this connects CR entropy to the Prime Number Theorem)")
print()
print("INTERPRETATION:")
print("  H2(G_n) measures how 'uniformly mixed' pairs of numbers are")
print("  (edge = share prime factor, no-edge = coprime)")
print("  Near n=1 where most pairs are coprime: H2 low")
print("  As n grows, edge density increases -> H2 varies")
print("  Mertens theorem: prod(1-1/p) ~ e^(-gamma)/log(n)")
print("  -> P(coprime pair) = 6/pi^2 ~ 0.608 (asymptotically)")
print("  -> H2 -> -0.608*log(0.608) - 0.392*log(0.392) = const = 0.666 bits??")
p_coprime = 6 / (math.pi**2)
p_noncoprime = 1 - p_coprime
H2_theory = -p_coprime * math.log(p_coprime) - p_noncoprime * math.log(p_noncoprime)
print(f"  Theoretical H2(n->inf) = {H2_theory:.4f} nats")
print(f"  -> H2/ln(n) -> 0 as n->inf (H2 is bounded!)")
