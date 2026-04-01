"""
BlackBox v3: Input-dependent gating (selective state update).
Key insight: WHICH information to keep depends on CURRENT input.
This is what Mamba does. Let's see if it closes the gap with transformer.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class BlackBoxV3(nn.Module):
    def __init__(self, vocab, d=128, n_inner=4):
        super().__init__()
        self.d = d; self.n_inner = n_inner
        self.embed = nn.Embedding(vocab, d)
        self.W = nn.Parameter(torch.randn(d, d) * 0.02)
        # INPUT-DEPENDENT gate: decides what to keep based on current token
        self.gate_net = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        # INPUT-DEPENDENT transform: modifies W based on input
        self.input_mod = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, vocab)

    def forward(self, input_ids):
        B, N = input_ids.shape
        embeds = self.embed(input_ids)
        state = torch.zeros(B, self.d, device=input_ids.device)
        outputs = []

        for pos in range(N):
            token = embeds[:, pos]

            for _ in range(self.n_inner):
                # Input-modulated transform
                mod = self.input_mod(token)  # [B, D]
                new_state = F.gelu(state @ self.W.T + mod)

                # Input-dependent gate: what to keep from old vs new
                gate = self.gate_net(torch.cat([state, token], dim=-1))
                state = gate * new_state + (1 - gate) * state

            outputs.append(state)

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
print("BLACKBOX v3 (input-dependent gate) vs Transformer")
print("="*55)

for name, make in [
    ("Transformer", lambda: Transformer(V,128,4,4,SEQ)),
    ("BB-v2-128", lambda: BlackBoxV3.__bases__[0].__subclasses__()[0](V,128,4) if False else None),
    ("BB-v3-128", lambda: BlackBoxV3(V,128,4)),
    ("BB-v3-256", lambda: BlackBoxV3(V,256,4)),
]:
    if name.startswith("BB-v2"): continue  # skip broken ref
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
        raccs=[]
        for r in range(3):
            m=(torch.arange(1000,device=DEVICE)%3==r)
            raccs.append((logits[m].argmax(-1)==tgt[m]).float().mean().item())
    print(f"  {name:14s}: PPL={ppl:.2f} acc={acc*100:.1f}% "
          f"[cp={raccs[0]*100:.0f}% inc={raccs[1]*100:.0f}% rep={raccs[2]*100:.0f}%] "
          f"p={np_:,} t={elapsed:.0f}s")
