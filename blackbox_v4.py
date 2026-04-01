"""
BlackBox v4: Multi-state. K parallel states, each tracking different info.
Input-dependent routing: which states to update for this token.
This is the most black-box architecture that can ACTUALLY compete.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class BlackBoxV4(nn.Module):
    def __init__(self, vocab, d=64, K=8, n_inner=3):
        super().__init__()
        self.d=d; self.K=K; self.n_inner=n_inner
        self.embed = nn.Embedding(vocab, d)
        # K separate transform matrices (one per state slot)
        self.Ws = nn.ParameterList([nn.Parameter(torch.randn(d,d)*0.02) for _ in range(K)])
        # Router: which states to update given current input
        self.router = nn.Linear(d, K)
        # Combiner: merge K states into one output
        self.combiner = nn.Linear(d*K, d)
        self.output = nn.Linear(d, vocab)

    def forward(self, input_ids):
        B, N = input_ids.shape
        embeds = self.embed(input_ids)
        # K parallel states
        states = [torch.zeros(B, self.d, device=input_ids.device) for _ in range(self.K)]
        outputs = []

        for pos in range(N):
            token = embeds[:, pos]  # [B, D]
            # Route: which states get updated by this token?
            route_weights = torch.softmax(self.router(token), dim=-1)  # [B, K]

            for _ in range(self.n_inner):
                for k in range(self.K):
                    new_s = F.gelu(states[k] @ self.Ws[k].T + token)
                    # Weighted update: high route weight = big update
                    w = route_weights[:, k:k+1]  # [B, 1]
                    states[k] = w * new_s + (1 - w) * states[k]

            # Combine all states for output
            combined = self.combiner(torch.cat(states, dim=-1))  # [B, D]
            outputs.append(combined)

        return self.output(torch.stack(outputs, dim=1))

class Transformer(nn.Module):
    def __init__(self, vocab, d=128, nh=4, nl=4, maxN=32):
        super().__init__()
        self.embed=nn.Embedding(vocab,d); self.pos=nn.Embedding(maxN,d)
        layer=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,nl); self.head=nn.Linear(d,vocab)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()
        return self.head(self.enc(x,mask=mask))

def gen_data(V=30, seq=16, batch=32):
    t=torch.randint(0,V,(batch,seq),device=DEVICE)
    for b in range(batch):
        r=b%3
        if r==0:
            for i in range(3,seq): t[b,i]=t[b,i-3]
        elif r==1:
            for i in range(1,seq): t[b,i]=(t[b,i-1]+1)%V
        else:
            for i in range(4,seq): t[b,i]=t[b,i%4]
    return t[:,:-1], t[:,1:]

V=30; SEQ=16; EP=100; BS=32
print("BLACKBOX v4 (multi-state + routing) vs Transformer")
print("="*55)

for name, make in [
    ("Transformer-4L", lambda: Transformer(V,128,4,4,SEQ)),
    ("BB-v4 K=4 d=64", lambda: BlackBoxV4(V,64,4,3)),
    ("BB-v4 K=8 d=64", lambda: BlackBoxV4(V,64,8,3)),
    ("BB-v4 K=16 d=64", lambda: BlackBoxV4(V,64,16,2)),
]:
    torch.manual_seed(42); model=make().to(DEVICE)
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
        raccs=[]
        for r in range(3):
            m=(torch.arange(1000,device=DEVICE)%3==r)
            raccs.append((logits[m].argmax(-1)==tgt[m]).float().mean().item())
    print(f"  {name:16s}: PPL={ppl:.2f} acc={acc*100:.1f}% "
          f"[cp={raccs[0]*100:.0f}% inc={raccs[1]*100:.0f}% rep={raccs[2]*100:.0f}%] "
          f"p={np_:,} t={elapsed:.0f}s")
