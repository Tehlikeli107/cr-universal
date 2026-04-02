"""
GRAVITY NET: Tokens are particles. They MOVE, ATTRACT, REPEL.

No attention. No FFT. No convolution.
Just: physics simulation on token embeddings.

Each token has:
  - position (embedding)
  - velocity (initially zero)
  - mass (learned from content — important tokens are heavier)

Forces:
  - Attraction between similar tokens (gravity)
  - Repulsion at very close range (prevent collapse)
  - Damping (prevent explosion)

After T timesteps of simulation: final positions = output.

This is GENUINELY NOVEL. Nobody has used N-body dynamics as
the core computation mechanism for sequence processing.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, math
DEVICE = torch.device('cuda')

class GravityLayer(nn.Module):
    """One step of gravitational dynamics."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        # Mass predictor: each token's "importance"
        self.mass_net = nn.Sequential(nn.Linear(d, d//4), nn.GELU(), nn.Linear(d//4, 1), nn.Softplus())
        # Force modifier: learned force law (not just 1/r^2)
        self.force_net = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Damping
        self.damping = nn.Parameter(torch.tensor(0.9))
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, pos, vel):
        """
        pos: [B, N, D] — current positions
        vel: [B, N, D] — current velocities
        Returns: new_pos, new_vel
        """
        B, N, D = pos.shape

        # Compute masses
        mass = self.mass_net(pos)  # [B, N, 1]

        # Pairwise displacement vectors
        # pos_i - pos_j for all pairs
        disp = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, D]
        dist = disp.norm(dim=-1, keepdim=True).clamp(min=0.01)  # [B, N, N, 1]

        # Direction
        direction = disp / dist  # [B, N, N, D]

        # Force magnitude: learned function of (distance, mass_i, mass_j)
        mass_i = mass.unsqueeze(2).expand(B, N, N, 1)
        mass_j = mass.unsqueeze(1).expand(B, N, N, 1)
        force_input = torch.cat([dist, mass_i, mass_j], dim=-1)  # [B, N, N, 3]
        force_mag = self.force_net(force_input)  # [B, N, N, 1]

        # Total force on each token: sum of forces from all others
        # Negative = attraction (toward other), positive = repulsion (away)
        force = (force_mag * direction).sum(dim=2)  # [B, N, D]

        # Causal: only feel force from PAST tokens
        causal_mask = torch.triu(torch.ones(N, N, device=DEVICE), 1).bool()
        force_causal = force_mag.clone()
        force_causal[:, :, :, :] = force_causal.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(-1), 0)
        force = (force_causal * direction).sum(dim=2)

        # Update velocity and position
        acc = force / (mass + 0.1)  # F = ma -> a = F/m
        new_vel = self.damping * vel + self.dt * acc
        new_pos = pos + self.dt * new_vel

        return new_pos, new_vel


class GravityNet(nn.Module):
    """Sequence model using gravitational dynamics."""
    def __init__(self, vocab, d, n_steps=6, max_seq=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos_enc = nn.Embedding(max_seq, d)
        # Multiple gravity steps
        self.steps = nn.ModuleList([GravityLayer(d) for _ in range(n_steps)])
        # FFN + Output
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        B, N = ids.shape
        pos = self.embed(ids) + self.pos_enc(torch.arange(N, device=ids.device))
        vel = torch.zeros_like(pos)

        # Simulate with FFN after each step
        for step in self.steps:
            pos, vel = step(pos, vel)
            # FFN for non-physical computation (arithmetic, etc.)
            pos = pos + self.ffn(self.ln(pos))

        return self.head(self.norm(pos))


class TF(nn.Module):
    def __init__(s,V,d,nh,nl,maxN):
        super().__init__()
        s.embed=nn.Embedding(V,d);s.pos=nn.Embedding(maxN,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl);s.head=nn.Linear(d,V)
    def forward(s,ids):
        B,N=ids.shape;x=s.embed(ids)+s.pos(torch.arange(N,device=ids.device))
        return s.head(s.enc(x,mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()))

def gen(V=20,seq=20,batch=48):
    t=torch.randint(0,V,(batch,seq),device=DEVICE);mid=seq//2
    for b in range(batch):
        for i in range(2,mid):t[b,i]=t[b,i-2]
        for i in range(max(mid,3),seq):t[b,i]=(t[b,i-1]+t[b,i-3])%V
    return t[:,:-1],t[:,1:]

V=20;SEQ=20;EP=150;D=48
print("GRAVITY NET vs TRANSFORMER")
print("="*50)
print("Tokens = particles. Attraction + repulsion = computation.\n")

for name,make in [
    ("Transformer", lambda:TF(V,D,4,3,SEQ)),
    ("Gravity 6step", lambda:GravityNet(V,D,6,SEQ)),
    ("Gravity 10step", lambda:GravityNet(V,D,10,SEQ)),
]:
    torch.manual_seed(42);model=make().to(DEVICE)
    np_=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    t0=time.time()
    for ep in range(EP):
        inp,tgt=gen(V,SEQ,48)
        loss=F.cross_entropy(model(inp).reshape(-1,V),tgt.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    elapsed=time.time()-t0
    with torch.no_grad():
        inp,tgt=gen(V,SEQ,1000)
        logits=model(inp)
        ppl=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).exp().item()
        acc=(logits.argmax(-1)==tgt).float().mean().item()
        mid=(SEQ-1)//2
        a1=(logits[:,:mid].argmax(-1)==tgt[:,:mid]).float().mean().item()
        a2=(logits[:,mid:].argmax(-1)==tgt[:,mid:]).float().mean().item()
    print(f"  {name:16s}: PPL={ppl:.2f} acc={acc*100:.1f}% [cp={a1*100:.0f}% sum={a2*100:.0f}%] p={np_:,} t={elapsed:.0f}s")
