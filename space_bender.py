"""
SPACE BENDER: Don't compute token relationships. BEND THE SPACE.

Instead of: "which token should I attend to?" (O(N^2))
Do: "change the geometry so relevant tokens are NEARBY" (O(N))

How: learned metric transformation. Each layer WARPS the embedding space.
After warping: simple LOCAL operations capture what was GLOBAL before.

This is OUTSIDE the pair/triple/N-ary hierarchy entirely.
No explicit token-token computation. Just space transformation + local op.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
DEVICE = torch.device('cuda')

class SpaceBendLayer(nn.Module):
    """
    Bend embedding space, then do LOCAL processing.

    Step 1: Learn a WARP matrix that transforms the space
            x_warped = x @ W_warp (this changes what "nearby" means)
    Step 2: Local operation: each token interacts with immediate neighbors only
            (cheap O(N) operation, but in WARPED space = captures global info)
    Step 3: Unwarp back to original space
    """
    def __init__(self, d, local_window=3):
        super().__init__()
        self.d = d
        self.w = local_window

        # WARP: learned space transformation (the "bending")
        self.warp = nn.Linear(d, d, bias=False)
        # Local mixing in warped space
        self.local_mix = nn.Linear(d * (2*local_window+1), d)
        # Unwarp
        self.unwarp = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape

        # Step 1: WARP the space
        x_warped = self.warp(x)  # [B, N, D]

        # Step 2: LOCAL operation in warped space
        # Each position gathers from local window
        padded = F.pad(x_warped, (0, 0, self.w, self.w))
        local_feats = []
        for offset in range(-self.w, self.w + 1):
            local_feats.append(padded[:, self.w + offset:self.w + offset + N])
        local_cat = torch.cat(local_feats, dim=-1)  # [B, N, D*(2w+1)]
        mixed = self.local_mix(local_cat)  # [B, N, D]

        # Step 3: UNWARP back
        x_out = self.unwarp(mixed)

        x = self.norm(x + x_out)
        x = self.norm2(x + self.ffn(x))
        return x


class InputDependentBend(nn.Module):
    """
    V2: The warp itself depends on INPUT (data-dependent geometry).
    Different inputs = different space bending = different "nearness."
    """
    def __init__(self, d, local_window=3):
        super().__init__()
        self.d = d; self.w = local_window
        # Generate warp matrix FROM input (data-dependent)
        self.warp_gen = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.local_mix = nn.Linear(d * (2*local_window+1), d)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        B, N, D = x.shape

        # Data-dependent warp: the bending depends on WHAT the tokens are
        warp_field = self.warp_gen(x)  # [B, N, D] — per-position warp
        x_warped = x * warp_field  # element-wise = per-dimension scaling

        # Local operation in warped space
        padded = F.pad(x_warped, (0,0,self.w,self.w))
        feats = torch.cat([padded[:, self.w+o:self.w+o+N] for o in range(-self.w, self.w+1)], dim=-1)

        # Causal: zero out future positions
        for o in range(1, self.w+1):
            feats[:, :, D*(self.w+o):D*(self.w+o+1)] = 0

        mixed = self.local_mix(feats)
        x = self.norm(x + mixed)
        x = self.norm2(x + self.ffn(x))
        return x


class SpaceBenderModel(nn.Module):
    def __init__(self, V, d, nl, maxN, use_input_dep=False):
        super().__init__()
        self.embed=nn.Embedding(V,d); self.pos=nn.Embedding(maxN,d)
        Layer = InputDependentBend if use_input_dep else SpaceBendLayer
        # Increasing local window with depth (like increasing receptive field)
        self.layers = nn.ModuleList([Layer(d, local_window=2+i) for i in range(nl)])
        self.head=nn.Linear(d,V)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        for l in self.layers: x=l(x)
        return self.head(x)

class TF(nn.Module):
    def __init__(s,V,d,nh,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl);s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        return s.head(s.enc(x,mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()))

def gen_mixed(V=20,seq=20,batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE);mid=seq//2
    for b in range(batch):
        for i in range(2,mid):t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq):t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;EP=150;D=48
print("SPACE BENDER vs TRANSFORMER")
print("="*50)
print("Space bending: O(N) local ops in warped space")
print("Transformer: O(N^2) attention\n")

for name,make in [
    ("Transformer", lambda: TF(V,D,4,3,SEQ)),
    ("SpaceBend", lambda: SpaceBenderModel(V,D,3,SEQ,False)),
    ("InputDepBend", lambda: SpaceBenderModel(V,D,3,SEQ,True)),
]:
    torch.manual_seed(42);model=make().to(DEVICE)
    np_=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    t0=time.time()
    for ep in range(EP):
        inp,tgt=gen_mixed(V,SEQ,48)
        loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    elapsed=time.time()-t0
    with torch.no_grad():
        inp,tgt=gen_mixed(V,SEQ,1000)
        logits=model(inp);ppl=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).exp().item()
        acc=(logits.argmax(-1)==tgt).float().mean().item()
        mid=(SEQ-1)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:14s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
