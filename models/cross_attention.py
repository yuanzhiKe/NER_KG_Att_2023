import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, x1, x2):
        """
        x1: shape (batch_size, seq_len1, d_model)
        x2: shape (batch_size, seq_len2, d_model)
        """
        # project to queries, keys and values
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        # compute attention weights and output
        attn_output, _ = self.multihead_attn(q, k, v)
        output = x1 + attn_output

        return output
