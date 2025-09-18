# torch>=2.0
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------

class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that runs in fp32 for stability even if inputs are fp16/bf16."""
    def forward(self, x):
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        hidden = int(hidden_mult * dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, kv_dim=None):
        """
        If kv_dim is provided, keys/values are projected from kv_dim -> dim.
        Otherwise, keys/values are projected from dim -> dim (self-attn case).
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.kv_dim = kv_dim if kv_dim is not None else dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(self.kv_dim, dim, bias=False)
        self.v_proj = nn.Linear(self.kv_dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, attn_mask=None):
        """
        x_q: (B, Nq, D) queries
        x_kv: (B, Nk, D_kv) keys/values source (defaults to x_q for self-attn)
        returns: (B, Nq, D)
        """
        if x_kv is None:
            x_kv = x_q
        B, Nq, D = x_q.shape
        Nk = x_kv.shape[1]

        q = self.q_proj(x_q).reshape(B, Nq, self.heads, D // self.heads).transpose(1, 2)  # (B, H, Nq, Dh)
        k = self.k_proj(x_kv).reshape(B, Nk, self.heads, D // self.heads).transpose(1, 2)  # (B, H, Nk, Dh)
        v = self.v_proj(x_kv).reshape(B, Nk, self.heads, D // self.heads).transpose(1, 2)  # (B, H, Nk, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, Nq, Dh)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.o_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNormFP32(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.ln2 = LayerNormFP32(dim)
        self.mlp = MLP(dim, hidden_mult=mlp_mult, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.ln_q = LayerNormFP32(q_dim)
        self.ln_kv = LayerNormFP32(kv_dim)
        self.attn = MultiHeadAttention(q_dim, heads=heads, dropout=dropout, kv_dim=kv_dim)
        self.ln_out = LayerNormFP32(q_dim)
        self.mlp = MLP(q_dim, hidden_mult=mlp_mult, dropout=dropout)

    def forward(self, q, kv, attn_mask=None):
        q = q + self.attn(self.ln_q(q), self.ln_kv(kv), attn_mask)
        q = q + self.mlp(self.ln_out(q))
        return q

# ----------------------------
# Perceiver-style encoder
# ----------------------------

class PerceiverPatchEncoder(nn.Module):
    """
    Encodes a set of patch embeddings (B, 50, 1024) into a single vector (B, out_dim).

    Steps:
      1) Optional learned positional enc for 50 patch positions (added to inputs).
      2) Cross-attn from L latent slots -> 50 tokens.
      3) K self-attn Transformer blocks on latent slots.
      4) Pool (mean or first) to 1 vector.
      5) Project to out_dim (e.g., 2048) for RSSM input.

    Recommended: small L (e.g., 16 or 32) to keep compute light.
    """
    def __init__(
        self,
        in_dim=1024,
        num_patches=50,
        latent_dim=512,
        num_latents=16,
        num_self_attn_blocks=4,
        heads=8,
        mlp_mult=4,
        dropout=0.0,
        out_dim=2048,
        use_positional_enc=True,
        pool="mean",  # "mean" or "cls"
    ):
        super().__init__()
        assert pool in ("mean", "cls")
        self.num_patches = num_patches
        self.use_positional_enc = use_positional_enc
        self.pool = pool

        # Project input patches to latent_dim (if needed)
        self.in_proj = nn.Linear(in_dim, latent_dim) if in_dim != latent_dim else nn.Identity()

        # Optional learned positional encodings for 50 positions
        if use_positional_enc:
            self.pos = nn.Parameter(torch.randn(1, num_patches, latent_dim) / math.sqrt(latent_dim))
        else:
            self.register_parameter("pos", None)

        # Learned latent array (L, latent_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) / math.sqrt(latent_dim))

        # One cross-attention to read from inputs
        self.cross = CrossAttentionBlock(
            q_dim=latent_dim, kv_dim=latent_dim, heads=heads, mlp_mult=mlp_mult, dropout=dropout
        )

        # Self-attention blocks over latents
        self.blocks = nn.ModuleList([
            TransformerBlock(latent_dim, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
            for _ in range(num_self_attn_blocks)
        ])

        # Optional "CLS" style: make first latent special
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) / math.sqrt(latent_dim))
        else:
            self.register_parameter("cls_token", None)

        # Final projection to out_dim (RSSM input)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def forward(self, patches):
        """
        patches: (B, Seq, 50, 1024) float
        returns: (B, Seq, out_dim)
        """
        B, S, N, D = patches.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        patches = patches.reshape(B * S, N, D)  # (B*Seq, 50, 1024)

        x = self.in_proj(patches)  # (B*Seq, 50, D)
        if self.use_positional_enc:
            x = x + self.pos

        # expand learned latents to batch
        latents = self.latents.expand(B * S, -1, -1)  # (B*Seq, L, latent_dim)

        # cross-attend latents to inputs
        latents = self.cross(latents, x)

        # self-attention on latents
        for blk in self.blocks:
            latents = blk(latents)

        if self.pool == "cls":
            # prepend CLS then self-attend a tiny bit more (optional); here we just take CLS projection
            cls = self.cls_token.expand(B * S, -1, -1)  # (B*Seq,1,C)
            # simple concat + linear pool could be used; we keep it minimal: use first latent as pooled
            pooled = latents[:, 0, :]
        else:
            pooled = latents.mean(dim=1)  # (B*Seq, C)

        output = self.out_proj(pooled)  # (B*Seq, out_dim)
        output = output.reshape(B, S, -1)  # (B, Seq, out_dim)
        return output


# ----------------------------
# Tiny usage example
# ----------------------------

if __name__ == "__main__":
    B = 8
    num_patches = 50
    patch_dim = 1024
    rssm_dim = 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(B, num_patches, patch_dim).to(device)

    encoder = PerceiverPatchEncoder(
        in_dim=patch_dim,
        num_patches=num_patches,
        latent_dim=512,
        num_latents=16,
        num_self_attn_blocks=4,
        heads=8,
        mlp_mult=4,
        dropout=0.0,
        out_dim=rssm_dim,
        use_positional_enc=True,
        pool="mean",  # "mean" or "cls"
    ).to(device)

    z = encoder(x)               # (B, rssm_dim)
    print(z.shape)
    