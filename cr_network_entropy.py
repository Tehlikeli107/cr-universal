"""
CR Network Entropy: Do Real Networks Deviate from Random?

HYPOTHESIS: Real-world networks (biological, social, infrastructure)
have DIFFERENT CR k-subgraph entropy than random graphs with same density.

If biological networks have H_3 > H_3(random): more diverse motif types
If social networks have H_3 < H_3(random): less diverse (more structured)

NEW MEASURE: CR Surprise Score S_k(G) = H_k(G) / H_k(G_random_same_density)
- S_k > 1: more diverse than random (over-represented rare motifs)
- S_k < 1: less diverse than random (dominant recurring patterns)
- S_k = 1: indistinguishable from random at scale k

DATASETS (all available in NetworkX):
- karate_club_graph: social network (34 nodes)
- florentine_families_graph: historical social network (15 nodes)
- les_miserables_graph: character co-appearance (77 nodes)
- petersen_graph: mathematical graph (10 nodes)
- icosahedral_graph: spatial graph (12 nodes)
- dodecahedral_graph: spatial graph (20 nodes)
- random graphs: Erdos-Renyi, Barabasi-Albert (for comparison)
"""
import numpy as np
from collections import Counter
from itertools import combinations
import networkx as nx
from math import log

def cr_k3_entropy(G, max_combos=10000):
    """H_3 = Shannon entropy of k=3 subgraph type distribution (0,1,2,3 edges)."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 3: return 0, Counter()

    combos = list(combinations(range(n), 3))
    if len(combos) > max_combos:
        idx = np.random.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in idx]

    type_counts = Counter()
    for i,j,k in combos:
        a,b,c = nodes[i], nodes[j], nodes[k]
        n_edges = int(G.has_edge(a,b)) + int(G.has_edge(a,c)) + int(G.has_edge(b,c))
        type_counts[n_edges] += 1

    total = sum(type_counts.values())
    H = 0
    for count in type_counts.values():
        p = count / total
        if p > 0:
            H -= p * log(p)
    return H, type_counts

def cr_k2_entropy(G):
    """H_2 = entropy of edge/non-edge distribution."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    total_pairs = n*(n-1)//2
    if total_pairs == 0: return 0
    p = m / total_pairs
    if p == 0 or p == 1: return 0
    return -p*log(p) - (1-p)*log(1-p)

def random_same_density(G, n_trials=10):
    """Average H_3 for random graphs with same density."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    total = n*(n-1)//2
    p = m / total if total > 0 else 0

    H_values = []
    for _ in range(n_trials):
        G_rand = nx.erdos_renyi_graph(n, p)
        H, _ = cr_k3_entropy(G_rand)
        H_values.append(H)
    return np.mean(H_values), np.std(H_values)

def analyze_network(name, G):
    """Full CR analysis."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2*m / (n*(n-1)) if n > 1 else 0
    cc = nx.average_clustering(G) if n > 2 else 0

    H2 = cr_k2_entropy(G)
    H3, dist3 = cr_k3_entropy(G)
    H3_rand, H3_rand_std = random_same_density(G, n_trials=5)

    S3 = H3 / H3_rand if H3_rand > 0 else 1
    surprise = (H3 - H3_rand) / max(H3_rand_std, 1e-6)  # z-score

    total3 = sum(dist3.values())
    p_triangle = dist3[3] / total3 if total3 > 0 else 0

    return {
        'name': name, 'n': n, 'm': m, 'density': density, 'cc': cc,
        'H2': H2, 'H3': H3, 'H3_rand': H3_rand, 'H3_rand_std': H3_rand_std,
        'S3': S3, 'z_score': surprise, 'p_triangle': p_triangle,
        'dist3': dist3, 'total3': total3
    }

print("=== CR Network Entropy: Real vs Random ===\n")
np.random.seed(42)

networks = {}

# Real-world networks from NetworkX
print("Building networks...", flush=True)
networks['Karate Club'] = nx.karate_club_graph()
networks['Florentine'] = nx.florentine_families_graph()
networks['Les Miserables'] = nx.les_miserables_graph()
networks['Petersen'] = nx.petersen_graph()
networks['Icosahedral'] = nx.icosahedral_graph()
networks['Dodecahedral'] = nx.dodecahedral_graph()

# Reference random graphs
networks['ER(n=34,p=0.14)'] = nx.erdos_renyi_graph(34, 0.14, seed=42)
networks['BA(n=34,m=2)'] = nx.barabasi_albert_graph(34, 2, seed=42)
networks['WS(n=34,k=4,b=0.1)'] = nx.watts_strogatz_graph(34, 4, 0.1, seed=42)

print("Analyzing networks...\n", flush=True)

results = []
for name, G in networks.items():
    r = analyze_network(name, G)
    results.append(r)

    # Distribution string
    total = r['total3']
    dist = r['dist3']
    dist_str = " ".join([f"{k}e:{dist[k]/total:.3f}" for k in sorted(dist.keys())])

    print(f"{name:25s}: n={r['n']:3d}, H3={r['H3']:.3f}, rand={r['H3_rand']:.3f}+-{r['H3_rand_std']:.3f}, S3={r['S3']:.3f}, z={r['z_score']:+.2f}, tri={r['p_triangle']:.4f}")

print()
print("=== SORTED BY SURPRISE SCORE (S3 = H3/H3_random) ===\n")
print(f"{'Network':25s}  {'S3':6s}  {'z-score':8s}  {'H3':6s}  {'p_tri':7s}  {'Interpretation'}")
print("-" * 80)
for r in sorted(results, key=lambda x: -x['S3']):
    interp = "MORE diverse than random" if r['S3'] > 1.05 else ("LESS diverse" if r['S3'] < 0.95 else "similar to random")
    print(f"  {r['name']:23s}  {r['S3']:6.3f}  {r['z_score']:+8.2f}  {r['H3']:6.3f}  {r['p_triangle']:7.4f}  {interp}")

print()
print("=== THEORY ===\n")

real_S3 = [r['S3'] for r in results if 'ER' not in r['name'] and 'BA' not in r['name'] and 'WS' not in r['name']]
rand_S3 = [r['S3'] for r in results if 'ER' in r['name'] or 'BA' in r['name'] or 'WS' in r['name']]

print(f"Real-world networks: mean S3 = {np.mean(real_S3):.3f} +/- {np.std(real_S3):.3f}")
print(f"Random networks:     mean S3 = {np.mean(rand_S3):.3f} +/- {np.std(rand_S3):.3f}")
print()

if np.mean(real_S3) > 1.1 * np.mean(rand_S3):
    print("CONFIRMED: Real networks have HIGHER CR entropy than random!")
    print("-> Real networks contain MORE diverse motif types than expected")
    print("-> 'Structural diversity' is a property of natural/social networks")
elif np.mean(real_S3) < 0.9 * np.mean(rand_S3):
    print("REVERSED: Real networks have LOWER CR entropy than random!")
    print("-> Real networks have FEWER motif types (more structured/specialized)")
else:
    print("Mixed: No universal CR entropy deviation from random for these networks")

print()
print("NOTE: The k=3 subgraph type distribution is related to the 'graphlet frequency'")
print("distribution (Przulj 2004). CR provides a principled way to choose k.")
print("The 'surprise score' S_k(G) may provide a new network complexity measure.")
