import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Part_Attention(nn.Module):
#     def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x_1, x_2, x_3):
#         B, N, C = x_1.shape # N = 1, C = 1152
#         # B,1,16,72
#         q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*72*1
#         k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*72*1
        
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         del q, k
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         v = self.linear_v(x_3).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*72*1
#         # x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B*1*128
#         x = (attn @ v).permute(0, 3, 1, 2) #B*1*16*72
#         x = x.reshape(B, N, C)
#         del v, attn
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x # B*1*128

# class FCAM2(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm): #0.1, 0.0, 0.1
#         super().__init__()
#         self.norm_1 = norm_layer(dim)
#         self.norm_2 = norm_layer(dim)
#         self.norm_3 = norm_layer(dim)

#         self.attn_1 = Part_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
#             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=dim*2, act_layer=act_layer, drop=drop)

#     def forward(self, x_1, x_2):
#         x_1 = x_1 + self.drop_path(self.attn_1(self.norm_1(x_2), self.norm_2(x_1), self.norm_3(x_1))) 
#         x_1 = x_1 + self.drop_path(self.mlp(self.norm(x_1))) 

#         return  x_1
    
class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) #None

    def forward(self, x_1):
        B, N, C = x_1.shape # N = 1, C = 512
        # B,1,16,32
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*32*1
        k = self.linear_k(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*32*1
        
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        del q, k
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        v = self.linear_v(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*80*1
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B*16*80*1 -> B*80*16*1 -> B*1*1280
        # x = (attn @ v).permute(0, 3, 2, 1) # B*16*80*1 -> B*1*80*16
        # x = x.reshape(B, N, C)
        del v, attn
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FSAM(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.norm_3 = norm_layer(dim)

        self.attn_1 = Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*2, act_layer=act_layer, drop=drop)

    def forward(self, x_1):
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm_1(x_1))) 
        x_1 = x_1 + self.drop_path(self.mlp(self.norm(x_1))) 

        return  x_1

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) #None

    def forward(self, x_1, x_2):
        B, N, C = x_1.shape # N = 1, C = 512
        # B,1,16,32
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*32*1
        k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*32*1
        
        attn = (q @ k.transpose(-2, -1)) * self.scale #B*16*32*32
        del q, k
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        v = self.linear_v(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*16*32*1
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B*16*32*1 -> B*32*16*1 -> B*1*512
        # x = (attn @ v).permute(0, 3, 2, 1) # B*16*32*1 -> B*1*32*16
        # x = x.reshape(B, N, C)
        del v, attn
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FCAM(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.norm_3 = norm_layer(dim)

        self.attn_1 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*2, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2):
        x = x_2 + self.drop_path(self.attn_1(self.norm_1(x_1), self.norm_2(x_2))) 
        x = x + self.drop_path(self.mlp(self.norm(x))) 

        return  x