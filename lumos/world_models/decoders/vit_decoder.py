import torch
from torch import nn
from lumos.world_models.encoders.vit_encoder import Attention, FeedForward

class Transformer(nn.Module):
    def __init__(self, out_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(out_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(out_dim, mlp_dim, out_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(self.norm1(x)) + x
            x = ff(self.norm2(x)) + x
        return x
    
class ViTDecoder(nn.Module):
    """
    ViT-style decoder that maps a latent vector of shape (B, in_dim)
    back to patch tokens of shape (B, 50, 1024).

    It mirrors the encoder in spirit:
      - Linear expand latent -> (B, 50, out_dim)
      - Add learnable positional embeddings
      - Process with a Transformer stack (self-attention + MLP)
      - Output (B, 50, out_dim)

    Args:
        num_patches: number of patch tokens (50)
        out_dim: dimensionality of each token (1024)
        depth, heads, dim_head, mlp_dim, dropout, emb_dropout: match encoder
        in_dim: dimension of the input latent vector (must match encoder out_dim)
        transformer_ctor: optional factory to supply your existing Transformer class:
            signature Transformer(out_dim, out_dim, depth, heads, dim_head, mlp_dim, dropout)
            If None, we import from the same module where this class is defined.
    """
    def __init__(
        self,
        *,
        num_frames: int = 1,
        num_patches: int = 50,
        out_dim: int = 1024,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        in_dim: int = 2048,   ):
        super().__init__()

        self.num_patches = num_patches
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames*num_patches, out_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        # Project latent vector -> initial token set of shape (_, num_patches, out_dim)
        self.latent_to_tokens = nn.Linear(in_dim, num_patches * out_dim)
        self.transformer = Transformer(
            out_dim=out_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Optional final norm to stabilize training (pairs nicely with MSE)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, in_dim)
        returns: (B, 50, 1024)
        """
        B, S, D = z.shape
        z = z.view(B * S, D)  # (B*Seq, in_dim)

        # (B*Seq, in_dim) -> (B*Seq, num_patches, out_dim)
        x = self.latent_to_tokens(z).view(B*S, self.num_patches, self.out_dim)
        x = x + self.pos_embedding[:, : self.num_patches]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.out_norm(x)
        x = x.view(B, S, self.num_patches, self.out_dim)

        return x