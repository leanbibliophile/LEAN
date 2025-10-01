"""
Adapted from https://github.com/lukemelas/PyTorch-Pretrained-ViT
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F

import loralib as lora
import models.lean_custom_layers as lean_lora

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class LoraMultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, lean=False, lora_rank=16, lora_flags=[True,False,True], lora_alpha=1, enable_stats=False, seq_len=0):
        super().__init__()
        self.lora_flags = lora_flags
        if lean:
            lora_model = lean_lora.LEANStatLinear
        else:
            lora_model = lean_lora.BaselineLinear
        if lora_flags[0]:    
            self.proj_q = lora_model(dim, dim, r=lora_rank, lora_alpha=lora_alpha, enable_stats=enable_stats, seq_len=seq_len) 
        else:
            self.proj_q = nn.Linear(dim, dim)
        if lora_flags[1]:
            self.proj_k = lora_model(dim, dim, r=lora_rank, lora_alpha=lora_alpha, enable_stats=enable_stats, seq_len=seq_len) 
        else:
            self.proj_k = nn.Linear(dim, dim)
        if lora_flags[2]:
            self.proj_v = lora_model(dim, dim, r=lora_rank, lora_alpha=lora_alpha, enable_stats=enable_stats, seq_len=seq_len) 
        else:
            self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        lora_weight_output = {}
        if self.lora_flags[0]:
            q, qw = self.proj_q(x)
            lora_weight_output['proj_q'] = qw
        else: 
            q = self.proj_q(x)
        if self.lora_flags[1]:
            k, kw = self.proj_q(x)
            lora_weight_output['proj_k'] = kw
        else:
            k = self.proj_k(x)
        if self.lora_flags[2]:
            v, vw = self.proj_v(x)
            lora_weight_output['proj_v'] = vw
        else:
            v = self.proj_v(x)

        #q0, k0, v0 = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, lora_weight_output


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class LoraBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, lean=False, lora_rank=16, lora_alpha=1, enable_stats=False, seq_len=0):

        super().__init__()
        self.attn = LoraMultiHeadedSelfAttention(dim, num_heads, dropout, lean=lean, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_stats=enable_stats, seq_len=seq_len)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h, lora_out = self.attn(self.norm1(x), mask)
        h = self.drop(self.proj(h))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, lora_out


class LoraTransformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, lean=False, lora_rank=16, lora_alpha=1, enable_stats=False, seq_len=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            LoraBlock(dim, num_heads, ff_dim, dropout, lean=lean, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_stats=enable_stats, seq_len=seq_len) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        lora_blocks_out = {}
        for idx, block in enumerate(self.blocks):
            x, lora_out = block(x, mask)
            lora_blocks_out[idx] = lora_out
        return x, lora_blocks_out
