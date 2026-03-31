import torch, traceback
DEVICE = torch.device('cuda')

def parse_graph6(line):
    line = line.strip()
    data = [ord(c)-63 for c in line]
    if data[0]<=62: n=data[0]; bs=1
    else: n=((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63); bs=4
    A = torch.zeros(n, n, device=DEVICE)
    bi=0
    for j in range(1,n):
        for i in range(j):
            bp=bs+bi//6; bw=5-(bi%6)
            if bp<len(data) and (data[bp]>>bw)&1: A[i,j]=A[j,i]=1
            bi+=1
    return A

g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
with open(g6_path) as f:
    line = f.readline()
adj = parse_graph6(line)
print(f"Graph: {adj.shape}")

try:
    from cr_gnn import compute_node_features, CRGNN
    feat = compute_node_features(adj)
    print(f"Features: {feat.shape}")

    torch.manual_seed(42)
    model = CRGNN(d_hidden=64, n_layers=3).to(DEVICE)
    emb = model(adj)
    print(f"Embedding: {emb.shape}")
    print("OK!")
except Exception as e:
    traceback.print_exc()
