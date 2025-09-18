import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lumos.world_models.encoders.transformer_encoder import TransformerBlock, CrossAttentionBlock

# ----------------------------
# Perceiver-style decoder to 50×1024
# ----------------------------

class PerceiverPatchDecoder(nn.Module):
    """
    Decodes a single vector (B, in_dim) back to 50×1024 patch embeddings.

    Steps:
      1) Project state vector -> latent slots (L, C).
      2) 50 learned query tokens cross-attend to those latents.
      3) Linear head -> 1024 dims per token.

    This mirrors Perceiver IO: queries (one per patch position) read from latent array.
    """
    def __init__(
        self,
        in_dim=2048,
        out_dim=1024,
        num_patches=50,
        latent_dim=512,
        num_latents=16,
        heads=8,
        mlp_mult=4,
        dropout=0.0,
        num_self_attn_blocks=2,
        use_positional_out=True,
    ):
        super().__init__()
        self.num_patches = num_patches

        # Project state -> latent slots
        self.to_latents = nn.Sequential(
            nn.Linear(in_dim, num_latents * latent_dim),
            nn.GELU(),
        )
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Optional self-attn on latents (improves capacity)
        self.latent_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
            for _ in range(num_self_attn_blocks)
        ])

        # Learned 50 query tokens that will produce per-patch outputs
        self.queries = nn.Parameter(torch.randn(1, num_patches, latent_dim) / math.sqrt(latent_dim))

        # Optional learned positional encodings added to queries
        if use_positional_out:
            self.pos_out = nn.Parameter(torch.randn(1, num_patches, latent_dim) / math.sqrt(latent_dim))
        else:
            self.register_parameter("pos_out", None)

        # Cross-attention from queries to latents
        self.cross = CrossAttentionBlock(
            q_dim=latent_dim, kv_dim=latent_dim, heads=heads, mlp_mult=mlp_mult, dropout=dropout
        )

        # Output head to patch embedding dimension
        self.out_head = nn.Linear(latent_dim, out_dim)

    def forward(self, state_vec):
        """
        state_vec: (B, Seq, in_dim) e.g., RSSM state or encoder output
        returns: (B, Seq, 50, 1024) reconstructed patch embeddings
        """
        B = state_vec.shape[0]
        S = state_vec.shape[1]
        latents = self.to_latents(state_vec).reshape(B * S, self.num_latents, self.latent_dim)  # (B*Seq, L, C)

        # optional self-attn on latents
        for blk in self.latent_blocks:
            latents = blk(latents)

        q = self.queries.expand(B * S, -1, -1)  # (B*Seq, 50, C)
        if hasattr(self, "pos_out") and self.pos_out is not None:
            q = q + self.pos_out

        dec = self.cross(q, latents)  # (B*Seq, 50, C)
        out = self.out_head(dec)      # (B*Seq, 50, 1024)
        out = out.reshape(B, S, self.num_patches, out.shape[-1])  # (B, Seq, 50, 1024)
        return out

if __name__ == "__main__":
    B = 8
    num_patches = 50
    patch_dim = 1024
    rssm_dim = 2048  # your target compact state size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(B, rssm_dim).to(device)

    decoder = PerceiverPatchDecoder(
        in_dim=rssm_dim,
        out_dim=patch_dim,
        num_patches=num_patches,
        latent_dim=512,
        num_latents=16,
        num_self_attn_blocks=2,
        heads=8,
        mlp_mult=4,
        dropout=0.0,
        use_positional_out=True,
    ).to(device)

    x_recon = decoder(z)        # (B, rssm_dim) -> feed this to your RSSM
    print(x_recon.shape)