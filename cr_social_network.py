"""
CR vs WL on Social Networks (COLLAB, no inherent node labels)

HYPOTHESIS (Label-k_min Conjecture):
  - Without node labels (c=1): CR needs large k to match WL
  - With degree-bin labels (c=bins): CR becomes more expressive relative to WL
  - Adding structural labels should help CR MORE than WL

COLLAB: 5000 scientific collaboration graphs, 3 classes
  - No node labels (all researchers same type)
  - Adding degree bins as structural labels

Test:
  1. WL, no labels (baseline)
  2. WL + degree-bin labels
  3. CR k=3, no labels
  4. CR k=3 + degree-bin labels
  5. CR+WL combined
"""
import numpy as np
import os
from collections import Counter
from itertools import combinations, permutations
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix, csr_matrix
import time

# ---- Load COLLAB ----
def load_tu_dataset(name, base_dir='.'):
    folder = os.path.join(base_dir, name)
    graph_labels = {}
    with open(os.path.join(folder, f'{name}_graph_labels.txt')) as f:
        for i, line in enumerate(f):
            v = line.strip()
            if v:
                graph_labels[i+1] = int(v)

    graph_indicator = {}
    with open(os.path.join(folder, f'{name}_graph_indicator.txt')) as f:
        for node_id, line in enumerate(f, 1):
            gid = int(line.strip())
            graph_indicator[node_id] = gid

    edges = []
    with open(os.path.join(folder, f'{name}_A.txt')) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                u, v = int(parts[0].strip()), int(parts[1].strip())
                edges.append((u, v))

    # Build graphs
    graphs_by_id = {}
    for node_id, gid in graph_indicator.items():
        if gid not in graphs_by_id:
            graphs_by_id[gid] = nx.Graph()
        graphs_by_id[gid].add_node(node_id)

    for u, v in edges:
        gid_u = graph_indicator.get(u)
        gid_v = graph_indicator.get(v)
        if gid_u == gid_v and gid_u is not None:
            graphs_by_id[gid_u].add_edge(u, v)

    # Sort by graph id
    gids = sorted(graph_labels.keys())
    G_list = [graphs_by_id.get(gid, nx.Graph()) for gid in gids]
    y = np.array([graph_labels[gid] for gid in gids])
    return G_list, y

print("Loading COLLAB...", flush=True)
G_list, y = load_tu_dataset('COLLAB')
print(f"COLLAB: {len(G_list)} graphs, {len(set(y))} classes")
print(f"Class dist: {Counter(y)}")
print(f"Avg nodes: {np.mean([G.number_of_nodes() for G in G_list]):.1f}")
print(f"Avg edges: {np.mean([G.number_of_edges() for G in G_list]):.1f}")
print()

# ---- Utility: add degree-bin labels ----
def add_degree_labels(G, bins=5):
    G2 = G.copy()
    degs = dict(G2.degree())
    if not degs:
        for n in G2.nodes():
            G2.nodes[n]['label'] = 1
        return G2
    max_d = max(degs.values()) + 1
    for n in G2.nodes():
        d = degs[n]
        bin_label = min(int(d * bins / max_d), bins-1) + 1
        G2.nodes[n]['label'] = bin_label
    return G2

# ---- WL Kernel ----
def wl_features(G_list, h=3, use_labels=False, n_bins=5):
    """Weisfeiler-Lehman subtree kernel features."""
    # Initial coloring
    colorings = []
    for G in G_list:
        if use_labels:
            c = {n: G.nodes[n].get('label', 1) for n in G.nodes()}
        else:
            c = {n: 1 for n in G.nodes()}  # all same
        colorings.append(c)

    all_feature_dicts = [{} for _ in G_list]
    color_counter = [0]
    color_map = {}

    def get_color(key):
        if key not in color_map:
            color_map[key] = color_counter[0]
            color_counter[0] += 1
        return color_map[key]

    for gi, (G, c) in enumerate(zip(G_list, colorings)):
        for n in G.nodes():
            k = ('init', c[n])
            col = get_color(k)
            all_feature_dicts[gi][col] = all_feature_dicts[gi].get(col, 0) + 1

    for iteration in range(h):
        new_colorings = []
        for gi, (G, c) in enumerate(zip(G_list, colorings)):
            new_c = {}
            for n in G.nodes():
                nbr_colors = tuple(sorted(c[nb] for nb in G.neighbors(n)))
                key = ('wl', iteration, c[n], nbr_colors)
                col = get_color(key)
                new_c[n] = col
                all_feature_dicts[gi][col] = all_feature_dicts[gi].get(col, 0) + 1
            new_colorings.append(new_c)
        colorings = new_colorings

    # Vectorize
    all_keys = sorted(set(k for d in all_feature_dicts for k in d))
    key_idx = {k: i for i, k in enumerate(all_keys)}
    X = np.zeros((len(G_list), len(all_keys)), dtype=np.float32)
    for gi, d in enumerate(all_feature_dicts):
        for k, v in d.items():
            X[gi, key_idx[k]] = v
    return X

# ---- CR Fingerprint (simplified for unlabeled/labeled) ----
def canonical_sub_labeled(nlist, G, use_labels=False):
    """Canonical k-subgraph with optional node labels."""
    n = len(nlist)
    nl = sorted(nlist)

    def node_label(v):
        if use_labels:
            return G.nodes[v].get('label', 1)
        return 1  # all same

    def make_adj(perm):
        mat = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(node_label(nl[perm[i]]))
                else:
                    # edge present?
                    edge_val = 1 if G.has_edge(nl[perm[i]], nl[perm[j]]) else 0
                    row.append(edge_val)
            mat.append(tuple(row))
        return tuple(mat)

    if n > 5:
        # Approximate: sorted atom types + edge tuple
        at = tuple(sorted(node_label(u) for u in nl))
        et = tuple(sorted((node_label(nl[i]), node_label(nl[j]), 1)
                          for i in range(n) for j in range(i+1, n)
                          if G.has_edge(nl[i], nl[j])))
        return (at, et)

    best = None
    for perm in permutations(range(n)):
        adj = make_adj(perm)
        if best is None or adj < best:
            best = adj
    return best

def cr_features(G_list, k=3, use_labels=False, max_combos=2000):
    """CR k-subgraph histogram features."""
    all_fps = []
    for gi, G in enumerate(G_list):
        nodes = list(G.nodes())
        if len(nodes) < k:
            all_fps.append(Counter())
            continue
        fp = Counter()
        combos = list(combinations(nodes, k))
        if len(combos) > max_combos:
            np.random.shuffle(combos)
            combos = combos[:max_combos]
        for sub in combos:
            canon = canonical_sub_labeled(list(sub), G, use_labels=use_labels)
            fp[canon] += 1
        all_fps.append(fp)

    # Vectorize
    all_keys = sorted(set(k for fp in all_fps for k in fp), key=str)
    key_idx = {k: i for i, k in enumerate(all_keys)}
    X = np.zeros((len(G_list), len(all_keys)), dtype=np.float32)
    for gi, fp in enumerate(all_fps):
        tot = sum(fp.values())
        for k, v in fp.items():
            X[gi, key_idx[k]] = v / max(tot, 1)
    return X

# ---- CV Evaluation ----
def evaluate(X, y, name, n_splits=10, seed=42):
    np.random.seed(seed)
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        clf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        scores.append(accuracy_score(y[te], pred))
    mean, std = np.mean(scores)*100, np.std(scores)*100
    print(f"  {name:40s}: {mean:.2f}% +/- {std:.2f}% (dim={X.shape[1]})")
    return mean, std

# Use subset for speed (COLLAB has 5000 graphs, can be slow)
MAX_GRAPHS = 1000
if len(G_list) > MAX_GRAPHS:
    print(f"Sampling {MAX_GRAPHS}/{len(G_list)} graphs for speed...")
    np.random.seed(42)
    idx = np.random.choice(len(G_list), MAX_GRAPHS, replace=False)
    G_sub = [G_list[i] for i in idx]
    y_sub = y[idx]
else:
    G_sub = G_list
    y_sub = y

print(f"Using {len(G_sub)} graphs for experiments")
print(f"Class dist: {Counter(y_sub)}")
print()

# Add degree labels to all graphs
print("Adding degree-bin labels (bins=5)...", flush=True)
G_labeled = [add_degree_labels(G, bins=5) for G in G_sub]

# ---- Feature extraction ----
print("Extracting WL features (no labels)...", flush=True)
t0 = time.time()
X_wl = wl_features(G_sub, h=3, use_labels=False)
print(f"  Done in {time.time()-t0:.1f}s, dim={X_wl.shape[1]}")

print("Extracting WL features (degree-bin labels)...", flush=True)
t0 = time.time()
X_wl_deg = wl_features(G_labeled, h=3, use_labels=True)
print(f"  Done in {time.time()-t0:.1f}s, dim={X_wl_deg.shape[1]}")

print("Extracting CR k=3 features (no labels)...", flush=True)
t0 = time.time()
X_cr = cr_features(G_sub, k=3, use_labels=False, max_combos=2000)
print(f"  Done in {time.time()-t0:.1f}s, dim={X_cr.shape[1]}")

print("Extracting CR k=3 features (degree-bin labels)...", flush=True)
t0 = time.time()
X_cr_deg = cr_features(G_labeled, k=3, use_labels=True, max_combos=2000)
print(f"  Done in {time.time()-t0:.1f}s, dim={X_cr_deg.shape[1]}")

# Combined features
X_cr_wl = np.hstack([X_cr_deg, X_wl_deg])

# ---- Evaluate ----
print("\n=== COLLAB Social Network Classification (10-fold CV) ===\n")
y_use = y_sub
results = {}
results['WL (no labels)'] = evaluate(X_wl, y_use, "WL k=3 (no labels)")
results['WL (degree-bin)'] = evaluate(X_wl_deg, y_use, "WL k=3 (degree-bin labels)")
results['CR k=3 (no labels)'] = evaluate(X_cr, y_use, "CR k=3 (no labels)")
results['CR k=3 (degree-bin)'] = evaluate(X_cr_deg, y_use, "CR k=3 (degree-bin labels)")
results['CR+WL (degree-bin)'] = evaluate(X_cr_wl, y_use, "CR+WL (degree-bin labels)")

print("\n=== SUMMARY ===\n")
print("Label-k_min Conjecture Test on Social Networks:")
print()
print(f"{'Method':40s} {'Acc':>8s} {'+/-':>6s}")
print("-" * 58)
for name, (mean, std) in results.items():
    print(f"  {name:38s}: {mean:6.2f}%  {std:5.2f}%")

print()
wl_base = results['WL (no labels)'][0]
wl_deg = results['WL (degree-bin)'][0]
cr_base = results['CR k=3 (no labels)'][0]
cr_deg = results['CR k=3 (degree-bin)'][0]

wl_gain = wl_deg - wl_base
cr_gain = cr_deg - cr_base

print(f"WL label gain:  {wl_gain:+.2f}% (no labels -> degree-bin)")
print(f"CR label gain:  {cr_gain:+.2f}% (no labels -> degree-bin)")
print()

if cr_gain > wl_gain:
    print("CONFIRMED: CR benefits MORE from adding labels than WL.")
    print("-> Label-k_min conjecture holds on social networks.")
    print(f"   CR gain={cr_gain:+.2f}% > WL gain={wl_gain:+.2f}%")
else:
    print("NOT confirmed: WL benefits more from labels than CR.")
    print(f"   CR gain={cr_gain:+.2f}%, WL gain={wl_gain:+.2f}%")

print()
if cr_deg > wl_deg:
    print(f"CR beats WL with degree-bin labels: {cr_deg:.2f}% vs {wl_deg:.2f}%")
else:
    print(f"WL beats CR even with degree-bin labels: {wl_deg:.2f}% vs {cr_deg:.2f}%")
