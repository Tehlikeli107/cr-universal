"""
LEARNED CONNECTION: Bağlantı fonksiyonunun kendisi öğrenilmiş.

Attention: bağlantı = dot_product(Q, K). İNSAN TASARIMI.
Bu: bağlantı = f(token_i, token_j) burada f = öğrenilmiş MLP.

f ne öğrenir? Bilmiyoruz. KARA KUTU. Belki dot product öğrenir,
belki DAHA İYİ bir şey öğrenir. Biz karar vermiyoruz.

Ayrıca: sadece İKİLİ değil, ÜÇLÜ bağlantı da deneyelim.
f(token_i, token_j, token_k) = üç token'ın BİRLİKTE ilişkisi.
Attention bunu YAPAMAZ.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class LearnedPairConnect(nn.Module):
    """Bağlantı = öğrenilmiş f(xi, xj). Dot product DEĞİL."""
    def __init__(self, d, d_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d*2, d*2), nn.GELU(),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, d_out))

    def forward(self, x):
        """x: [B, N, D]. Returns: [B, N, d_out] aggregated."""
        B, N, D = x.shape
        # For each pair (i, j): compute f(xi, xj)
        # Efficient: use broadcasting
        xi = x.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        xj = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        pair_input = torch.cat([xi, xj], dim=-1)  # [B, N, N, 2D]

        # Learned connection strength + value
        pair_out = self.f(pair_input)  # [B, N, N, d_out]

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=DEVICE), 1).bool()
        pair_out = pair_out.masked_fill(mask.unsqueeze(0).unsqueeze(-1), 0)

        # Aggregate: softmax-weighted sum (like attention but learned weights)
        weights = pair_out.norm(dim=-1, keepdim=True)  # [B, N, N, 1]
        weights = weights.masked_fill(mask.unsqueeze(0).unsqueeze(-1), -1e9)
        weights = F.softmax(weights.squeeze(-1), dim=-1).unsqueeze(-1)  # [B, N, N, 1]

        return (weights * pair_out).sum(dim=2)  # [B, N, d_out]


class LearnedTripleConnect(nn.Module):
    """ÜÇLÜ bağlantı: f(xi, xj, xk). Attention'ın YAPAMAYACAĞI şey."""
    def __init__(self, d, d_out, n_sample=8):
        super().__init__()
        self.n_sample = n_sample  # sample triplets (O(N^3) too expensive)
        self.f = nn.Sequential(
            nn.Linear(d*3, d*2), nn.GELU(),
            nn.Linear(d*2, d_out))

    def forward(self, x):
        B, N, D = x.shape
        S = min(self.n_sample, N)

        # Sample S random (j, k) pairs for each i
        j_idx = torch.randint(N, (B, N, S), device=DEVICE)
        k_idx = torch.randint(N, (B, N, S), device=DEVICE)

        # Gather
        xj = x.gather(1, j_idx.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xk = x.gather(1, k_idx.reshape(B,-1,1).expand(-1,-1,D)).reshape(B,N,S,D)
        xi = x.unsqueeze(2).expand(B,N,S,D)

        # Compute f(xi, xj, xk)
        triple = torch.cat([xi, xj, xk], dim=-1)  # [B, N, S, 3D]
        out = self.f(triple)  # [B, N, S, d_out]

        return out.mean(dim=2)  # [B, N, d_out]


class LearnedConnectModel(nn.Module):
    """Full model: embed -> learned_pair_connect -> ffn -> repeat."""
    def __init__(self, vocab, d=64, n_layers=2, max_seq=32, use_triple=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if use_triple:
                self.layers.append(nn.ModuleDict({
                    'connect': LearnedTripleConnect(d, d, n_sample=6),
                    'norm1': nn.LayerNorm(d),
                    'ffn': nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                    'norm2': nn.LayerNorm(d),
                }))
            else:
                self.layers.append(nn.ModuleDict({
                    'connect': LearnedPairConnect(d, d),
                    'norm1': nn.LayerNorm(d),
                    'ffn': nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)),
                    'norm2': nn.LayerNorm(d),
                }))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        B, N = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(N, device=ids.device))
        for layer in self.layers:
            x = layer['norm1'](x + layer['connect'](x))
            x = layer['norm2'](x + layer['ffn'](x))
        return self.head(x)

class Transformer(nn.Module):
    def __init__(self, vocab, d=64, nh=4, nl=2, max_seq=32):
        super().__init__()
        self.embed=nn.Embedding(vocab,d); self.pos=nn.Embedding(max_seq,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.head=nn.Linear(d,vocab)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()
        return self.head(self.enc(x,mask=mask))

def gen_data(V=20, seq=24, batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE)
    mid=seq//2
    for b in range(batch):
        for i in range(2,mid): t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq): t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20; SEQ=20; EP=150; BS=48; D=48

print("LEARNED CONNECTION vs TRANSFORMER vs TRIPLE CONNECTION")
print("="*55)
print(f"V={V} seq={SEQ} ep={EP} d={D}\n")

for name, make in [
    ("Transformer", lambda: Transformer(V,D,4,2,SEQ)),
    ("LearnedPair", lambda: LearnedConnectModel(V,D,2,SEQ,False)),
    ("LearnedTriple", lambda: LearnedConnectModel(V,D,2,SEQ,True)),
]:
    torch.manual_seed(42)
    model=make().to(DEVICE)
    np_=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    t0=time.time()
    for ep in range(EP):
        inp,tgt=gen_data(V,SEQ,BS)
        loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    elapsed=time.time()-t0
    with torch.no_grad():
        inp,tgt=gen_data(V,SEQ,1000)
        logits=model(inp)
        ppl=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).exp().item()
        acc=(logits.argmax(-1)==tgt).float().mean().item()
        mid=(SEQ-1)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:14s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
