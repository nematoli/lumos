# adapted from DINO-WM
import torch
from torch import nn
from einops import rearrange, repeat

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 50

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.out_norm = nn.LayerNorm(in_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(in_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(in_dim, mlp_dim, in_dim, dropout = dropout)
            ]))
        self.final_layer = nn.ModuleList([
                Attention(in_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(in_dim, mlp_dim, out_dim, dropout = dropout)
            ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(self.norm1(x)) + x
            x = ff(self.norm2(x)) + x

        x = self.final_layer[0](self.out_norm(x)) + x
        x = self.final_layer[1](self.out_norm(x))
        return x
    
class ViTEncoder(nn.Module):
    """
    ViT-style encode that maps a latent patches of shape (Seq, B, Num_Patches, Patch_Dim)
    to a flattened vector of shape (Seq, B, Num_Patches * Out_Dim).

    Args:
        num_patches: number of patch tokens (50)
        in_dim: dimensionality of each token (1024)
        depth: number of transformer blocks (6)
        heads: number of attention heads (8)
        mlp_dim: hidden dimension of the MLP in each transformer block (2048)
        out_dim: output dimension of the final token (40)
        dim_head: dimension of each attention head (64)
        dropout: dropout rate (0.1)
        emb_dropout: dropout rate for the input embedding (0.1)
    """
    def __init__(self, *, num_patches, num_frames, in_dim, depth, heads, mlp_dim, out_dim, dim_head=64, dropout=0., emb_dropout=0.,):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), in_dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(in_dim, out_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x): # x: (Seq, B, 50, 1024)
        S, B, N, D = x.shape
        x = x.reshape(S*B, N, D)  # (Seq*B, 50, 1024)
        x = x + self.pos_embedding[:, :N]
        x = self.dropout(x) 
        x = self.transformer(x) # (Seq*B, 50, out_dim)
        x = x.view(S, B, -1) # (Seq, B, 50*out_dim)
        return x