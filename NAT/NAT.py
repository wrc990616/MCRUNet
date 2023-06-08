import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

class NeighborhoodAttention(
    nn.Module):  # It can only use static size as input,but you can define a new input size if you wish.
    def __init__(self, input_size, dim, num_heads, window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert window_size % 2 == 1, 'windowsize must be odd.'
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size // 2)
        self.relative_bias = nn.Parameter(torch.zeros((2 * self.window_size - 1) ** 2, num_heads))
        trunc_normal_(self.relative_bias, std=.02)
        self.idx_h = torch.arange(0, window_size)
        self.idx_w = torch.arange(0, window_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2 * self.window_size - 1)) + self.idx_w).view(-1)
        self.set_input_size(input_size)

    def forward(self, x):
        x = self.attention(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def attention(self, x):
        B, C, H, W = x.shape
        assert H >= self.window_size and W >= self.window_size, 'input size must not be smaller than window size'
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q.unsqueeze(3) @ k[:, :, self.attn_idx].transpose(-1, -2)  # B,nh,L,1,K^2
        attn = attn + self.relative_bias[self.bias_idx].permute(2, 0, 1).unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v[:, :, self.attn_idx]).squeeze(3).transpose(-1, -2).contiguous().view(B, C, H, W)
        return x

    def get_bias_idx(self, H, W):
        num_repeat_h = torch.ones(self.window_size, dtype=torch.long)
        num_repeat_w = torch.ones(self.window_size, dtype=torch.long)
        num_repeat_h[self.window_size // 2] = H - (self.window_size - 1)
        num_repeat_w[self.window_size // 2] = W - (self.window_size - 1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
                    2 * self.window_size - 1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        return bias_idx.view(-1, self.window_size ** 2)

    def get_attn_idx(self, H, W):
        H_ = H - (self.window_size - 1)
        W_ = W - (self.window_size - 1)
        attn_idx = torch.arange(0, H_ * W_, dtype=torch.float).view(1, 1, H_, W_)
        attn_idx = self.pad_idx(attn_idx).view(-1).type(torch.long)
        attn_idx = self.get_unfold_idx(H, W)[attn_idx]
        return attn_idx

    def get_unfold_idx(self, H, W):
        H_ = H - (self.window_size - 1)
        W_ = W - (self.window_size - 1)
        h_idx = torch.arange(W_).repeat(H_)
        w_idx = torch.arange(H_).repeat_interleave(W_) * W
        k_idx_1 = torch.arange(self.window_size).repeat(self.window_size)
        k_idx_2 = torch.arange(self.window_size).repeat_interleave(self.window_size) * W
        k_idx = k_idx_1 + k_idx_2
        hw_idx = h_idx + w_idx
        unfold_idx = hw_idx[:, None] + k_idx
        return unfold_idx

    def set_input_size(self, input_size):
        H, W = input_size
        self.H, self.W = H, W
        assert H >= self.window_size and W >= self.window_size, 'input size must not be smaller than window size'
        attn_idx = self.get_attn_idx(H, W)
        bias_idx = self.get_bias_idx(H, W)
        self.register_buffer("attn_idx", attn_idx)
        self.register_buffer("bias_idx", bias_idx)
if __name__ == '__main__':

	net =NeighborhoodAttention(input_size=(7,7),dim=64,num_heads=2)
	x = torch.randn(1, 64, 7 ,7)
	y = net(x)
	print(x.size())