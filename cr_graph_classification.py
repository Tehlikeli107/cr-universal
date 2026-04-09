"""
CR Fingerprints for Graph Classification (TUDatasets)

THEORY: Counting Revolution shows CR induced subgraph histograms are
strictly stronger than WL hierarchy for graph isomorphism.
PREDICTION: CR should beat WL kernel and WL-based GNNs on graph classification.

Datasets: MUTAG, PROTEINS from TU Dortmund benchmark
Methods: WL subtree kernel (standard), CR k=3, CR k=4

Published results (10-fold CV accuracy):
  WL kernel:   MUTAG=90.4%, PROTEINS=75.0%
  GIN:         MUTAG=89.0%, PROTEINS=76.2%
  DiffPool:    MUTAG=79.6%, PROTEINS=76.3%
  CW Network:  MUTAG=96.5% (topological)
"""
import numpy as np
import urllib.request, zipfile, io, os
from collections import Counter, defaultdict
from itertools import combinations, permutations
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

def download_tudataset(name, cache_dir=r'C:\Users\salih\Desktop\cr-universal\tudatasets'):
    """Download TU Dortmund dataset."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, name)
    if os.path.exists(local_path):
        print(f"  Using cached {name}")
        return local_path

    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    print(f"  Downloading {name}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Python'})
        r = urllib.request.urlopen(req, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(cache_dir)
        print(f"  Extracted to {local_path}")
        return local_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

def load_tudataset(path, name):
    """Load TU dataset into list of networkx graphs."""
    def read_file(suffix):
        fpath = os.path.join(path, f"{name}_{suffix}.txt")
        if not os.path.exists(fpath): return None
        with open(fpath) as f:
            return [line.strip() for line in f if line.strip()]

    # Graph indicator: which graph does each node belong to?
    graph_indicator = [int(x) for x in read_file('graph_indicator')]
    graph_labels_raw = read_file('graph_labels')
    graph_labels = [int(x) for x in graph_labels_raw]

    # Build node -> graph mapping
    n_nodes = len(graph_indicator)
    n_graphs = max(graph_indicator)

    # Node labels (optional)
    node_labels_raw = read_file('node_labels')
    node_labels = [int(x) for x in node_labels_raw] if node_labels_raw else [1] * n_nodes

    # Edge list (1-indexed)
    edges_raw = read_file('A')
    edges = [(int(e.split(',')[0]), int(e.split(',')[1])) for e in edges_raw]

    # Edge labels (optional)
    edge_labels_raw = read_file('edge_labels')
    edge_labels = [int(x) for x in edge_labels_raw] if edge_labels_raw else [1] * len(edges)

    # Build graphs
    graphs = [nx.Graph() for _ in range(n_graphs)]
    labels = graph_labels

    for node_id, (graph_id, node_label) in enumerate(zip(graph_indicator, node_labels), start=1):
        graphs[graph_id - 1].add_node(node_id, label=node_label)

    for (u, v), el in zip(edges, edge_labels):
        gu = graph_indicator[u - 1] - 1
        graphs[gu].add_edge(u, v, label=el)

    return graphs, labels

def canonical_sub(node_labels, edge_dict, nlist):
    """Canonical form of induced k-subgraph with typed nodes and edges."""
    n = len(nlist); nl = sorted(nlist)
    if n <= 6:
        best = None
        for perm in permutations(range(n)):
            mat = tuple(
                tuple(
                    node_labels[nl[perm[i]]] if i == j
                    else edge_dict.get((nl[perm[i]], nl[perm[j]]), 0)
                    for j in range(n)
                )
                for i in range(n)
            )
            if best is None or mat < best:
                best = mat
        return best
    # Approximate for large n
    at = tuple(sorted(node_labels[u] for u in nl))
    et = tuple(sorted(
        (node_labels[nl[i]], node_labels[nl[j]], edge_dict.get((nl[i], nl[j]), 0))
        for i in range(n) for j in range(i+1, n)
        if edge_dict.get((nl[i], nl[j]), 0) > 0
    ))
    return (at, et)

def cr_fingerprint_graph(G, k=3, max_combos=3000):
    """CR k-subgraph fingerprint for a networkx graph."""
    # Build typed graph
    node_labels = {n: G.nodes[n].get('label', 1) for n in G.nodes()}
    edge_dict = {}
    for u, v, d in G.edges(data=True):
        el = d.get('label', 1)
        edge_dict[(u, v)] = el
        edge_dict[(v, u)] = el

    nodes = list(G.nodes())
    if len(nodes) < k:
        return Counter()

    c = Counter()
    combos = list(combinations(nodes, k))
    if len(combos) > max_combos:
        np.random.shuffle(combos)
        combos = combos[:max_combos]

    for sub in combos:
        sub_labels = {n: node_labels[n] for n in sub}
        sub_edges = {}
        for i in range(len(sub)):
            for j in range(i+1, len(sub)):
                u, v = sub[i], sub[j]
                if edge_dict.get((u, v), 0) > 0:
                    sub_edges[(u, v)] = edge_dict[(u, v)]
                    sub_edges[(v, u)] = edge_dict[(v, u)]
        c[canonical_sub(sub_labels, sub_edges, list(sub))] += 1
    return c

def wl_fingerprint(G, h=3):
    """WL subtree kernel feature (h iterations of label refinement)."""
    # Initialize with node labels
    labels = {n: G.nodes[n].get('label', 1) for n in G.nodes()}
    all_labels = Counter()
    for v, l in labels.items():
        all_labels[(0, l)] += 1

    label_map = {}
    next_label = max(labels.values()) + 1

    for it in range(1, h + 1):
        new_labels = {}
        for v in G.nodes():
            neighbor_labels = sorted(labels[u] for u in G.neighbors(v))
            signature = (labels[v], tuple(neighbor_labels))
            if signature not in label_map:
                label_map[signature] = next_label
                next_label += 1
            new_labels[v] = label_map[signature]
        labels = new_labels
        for v, l in labels.items():
            all_labels[(it, l)] += 1

    return all_labels

# =========================================================
print("=== CR vs WL: Graph Classification (TUDatasets) ===\n")
print("Theory: CR induced subgraph histogram is strictly stronger than WL hierarchy")
print("Prediction: CR >= WL kernel for graph classification accuracy\n")

published_results = {
    'MUTAG': {'WL kernel': 90.4, 'GIN': 89.0, 'DiffPool': 79.6},
    'PROTEINS': {'WL kernel': 75.0, 'GIN': 76.2, 'DiffPool': 76.3},
    'IMDB-B': {'WL kernel': 73.8, 'GIN': 75.1},
}

datasets_to_test = ['MUTAG', 'PROTEINS']  # Start with small ones

for dataset_name in datasets_to_test:
    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*55}")
    print(f"Published: {published_results.get(dataset_name, {})}")
    print()

    path = download_tudataset(dataset_name)
    if path is None:
        print(f"  Skipping {dataset_name} (download failed)")
        continue

    graphs, labels = load_tudataset(path, dataset_name)
    y = np.array(labels)
    print(f"  Loaded: {len(graphs)} graphs, {len(set(labels))} classes")
    n_nodes_list = [G.number_of_nodes() for G in graphs]
    print(f"  Avg nodes: {np.mean(n_nodes_list):.1f}, max: {max(n_nodes_list)}")
    print()

    # Compute WL fingerprints (h=3)
    print("  Computing WL h=3...", flush=True)
    wl_fps = [wl_fingerprint(G, h=3) for G in graphs]
    wl_vocab = {t:i for i,t in enumerate(set(t for fp in wl_fps for t in fp))}
    Xwl = np.zeros((len(graphs), len(wl_vocab)))
    for i, fp in enumerate(wl_fps):
        tot = sum(fp.values())
        for t, c in fp.items():
            if t in wl_vocab: Xwl[i, wl_vocab[t]] = c / max(tot, 1)

    # Compute CR k=3 fingerprints
    print("  Computing CR k=3...", flush=True)
    cr3_fps = [cr_fingerprint_graph(G, k=3) for G in graphs]
    cr3_vocab = {t:i for i,t in enumerate(set(t for fp in cr3_fps for t in fp))}
    Xcr3 = np.zeros((len(graphs), len(cr3_vocab)))
    for i, fp in enumerate(cr3_fps):
        tot = sum(fp.values())
        for t, c in fp.items():
            if t in cr3_vocab: Xcr3[i, cr3_vocab[t]] = c / max(tot, 1)

    # Compute CR k=4 fingerprints (if n <= 40 avg)
    if np.mean(n_nodes_list) <= 50:
        print("  Computing CR k=4...", flush=True)
        cr4_fps = [cr_fingerprint_graph(G, k=4, max_combos=2000) for G in graphs]
        cr4_vocab = {t:i for i,t in enumerate(set(t for fp in cr4_fps for t in fp))}
        Xcr4 = np.zeros((len(graphs), len(cr4_vocab)))
        for i, fp in enumerate(cr4_fps):
            tot = sum(fp.values())
            for t, c in fp.items():
                if t in cr4_vocab: Xcr4[i, cr4_vocab[t]] = c / max(tot, 1)
        has_cr4 = True
    else:
        has_cr4 = False
        print("  Skipping CR k=4 (large graphs)")

    print(f"\n  WL vocab size: {len(wl_vocab)}")
    print(f"  CR k=3 types: {len(cr3_vocab)}")
    if has_cr4: print(f"  CR k=4 types: {len(cr4_vocab)}")
    print()

    # 10-fold CV (same as published)
    configs = [
        ("WL kernel (h=3)", Xwl),
        ("CR k=3", Xcr3),
    ]
    if has_cr4:
        configs.append(("CR k=4", Xcr4))
        configs.append(("CR k=3+4", np.hstack([Xcr3, Xcr4])))

    print(f"  10-fold CV accuracy (mean +- std):")
    print(f"  {'Method':>20s}  Accuracy")
    print("  " + "-" * 35)
    results = {}
    for name, X in configs:
        accs = []
        for seed in range(5):
            cv = StratifiedKFold(10, shuffle=True, random_state=seed)
            # SVM with RBF works well for kernel-style features
            clf = SVC(kernel='rbf', C=10, gamma='scale')
            scores = cross_val_score(clf, normalize(X), y, cv=cv, scoring='accuracy')
            accs.append(scores.mean() * 100)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[name] = mean_acc
        print(f"  {name:>20s}: {mean_acc:.2f}% +- {std_acc:.2f}%")

    print()
    print(f"  Published WL kernel: {published_results.get(dataset_name, {}).get('WL kernel', '?')}%")
    print(f"  Published GIN:       {published_results.get(dataset_name, {}).get('GIN', '?')}%")
    wl_acc = results.get('WL kernel (h=3)', 0)
    cr3_acc = results.get('CR k=3', 0)
    print(f"  Our WL:   {wl_acc:.2f}%")
    print(f"  Our CR k=3: {cr3_acc:.2f}%  (vs WL: {cr3_acc-wl_acc:+.2f}%)")
    if has_cr4:
        cr4_acc = results.get('CR k=4', 0)
        cr34_acc = results.get('CR k=3+4', 0)
        print(f"  Our CR k=4: {cr4_acc:.2f}%")
        print(f"  Our CR k=3+4: {cr34_acc:.2f}%")

print("\n\n=== THEORETICAL SUMMARY ===")
print()
print("Counting Revolution predicts: CR k=n-3 CAN distinguish ALL non-isomorphic graphs")
print("WL kernel can NOT distinguish some non-isomorphic graphs (e.g. Shrikhande vs Rook(4))")
print()
print("For graph classification: CR captures more structural information than WL")
print("-> CR should have higher expressivity -> better classification accuracy")
print()
print("Note: k=3 may not reach the theoretical k_min = n-3 for all graphs,")
print("but captures more information than WL (which has limited expressivity)")
