import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import os
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from .video_diffusion_pytorch_text import tokenize, bert_embed, BERT_MODEL_DIM
# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    #num=16, divisor=32
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        # print()

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias
        
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet',
        image_size=16,
        cond_channels =0
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        # print(cond_dim) #None
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        # print(self.has_cond, use_bert_text_cond, cond_dim, BERT_MODEL_DIM)
        #True True 768 768

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)

        #[16, 2, 7, 16, 16]
        self.null_src_cond_emb = nn.Parameter(torch.zeros(1, 2, 7, image_size, image_size)) if cond_channels == 0 else nn.Parameter(torch.zeros(1, cond_channels, 7, image_size, image_size))
        

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = ResnetBlock
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        src_cond_scale = 2., 
        **kwargs
    ):
        ''' logits = self.forward(*args, null_cond_prob = 0., **kwargs)     # 没有条件指导的概率=0,生成有条件指导的视频
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)   # 没有条件指导的概率=1,生成没有条件指导的视频 '''
        
        
        ''' #nellie
        logits = self.forward(*args, null_cond_prob = 0., null_src_cond_prob=0., **kwargs)     # 没有条件指导的概率=0,生成有条件指导的视频
        if cond_scale == 1 or not self.has_cond:
            return logits
        null_logits = self.forward(*args, null_cond_prob = 0., null_src_cond_prob=1., **kwargs)   # 没有条件指导的概率=1,生成没有条件指导的视频
        return null_logits + (logits - null_logits) * cond_scale '''

        #nellie-2
        #No condition -> most diversity
        
        null_logits = self.forward(*args, null_cond_prob = 1., null_src_cond_prob=1., **kwargs)     #100%没有条件，增加多样性
        #add condition-source -> reduce diversity
        logits1 = self.forward(*args, null_cond_prob = 1., null_src_cond_prob=0., **kwargs)
        #add condition-source+text -> reduce diversity 
        logits2 = self.forward(*args, null_cond_prob = 0., null_src_cond_prob=0., **kwargs)
        
        return (1-src_cond_scale) * null_logits + (src_cond_scale - cond_scale) * logits1 + cond_scale * logits2

        
        logits_text = self.forward(*args, null_cond_prob = 0., null_src_cond_prob=0., **kwargs)     # 没有条件指导的概率=0,100%有条件
       
        if cond_scale == 1 or not self.has_cond:
            return logits
        null_logits = self.forward(*args, null_cond_prob = 0., null_src_cond_prob=1., **kwargs)   # 没有条件指导的概率=1，生成没有条件指导的视频
        return null_logits + (logits - null_logits) * cond_scale
    


    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        
        src_cond = None,
        null_src_cond_prob = 0.,
        
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        # print("\n\n\n",null_src_cond_prob, null_cond_prob)  #0.7 0.0
        # assert 1>222
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device)) #当 prob = 0 时，会返回一个全为 False 的布尔张量。

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        # time_rel_pos_bias 的形状为 [8, 7, 7]，意味着为 8 个注意力头生成了 7 个时间步之间的相对位置偏置矩阵。每个矩阵的维度是 7 x 7，代表每个时间步与其他时间步的相对位置偏置。
        # print(time_rel_pos_bias.shape,"~~~~~~~~~~x~~~~~~~~", x.shape) #[8, 7, 7]  [40, 22, 7, 16, 16]
        
        # print(src_cond.shape)   #[48, 2, 7, 64, 64]
        # assert 3>444
        if exists(src_cond):
            batch, device = x.shape[0], x.device
            # print(null_cond_prob) #=0
            mask = prob_mask_like((batch,), null_src_cond_prob, device = device) ##当 null_cond_prob = 0 时，会返回一个全为 False 的布尔张量。
            
            #torch.where(condition, x, y) 是一个条件选择函数，用于根据 condition 张量的布尔值选择 x 或 `y。
            #如果 condition 对应位置的值为 True，选择 x 的值。 如果 condition 对应位置的值为 False，选择 y 的值。
            # print(self.null_cond_emb.shape, self.null_cond_emb) #torch.Size([1, 768]) 
            # print(mask.shape, self.null_src_cond_emb.shape, src_cond.shape) #torch.Size([40]) torch.Size([1, 2, 7, 16, 16]) torch.Size([40, 2, 7, 16, 16])
            
            src_cond = torch.where(rearrange(mask, 'b -> b 1 1 1 1'), self.null_src_cond_emb, src_cond)
            x = torch.cat((x, src_cond), dim = 1)



        
        x = self.init_conv(x)  #kernal-size: (1, init_kernel_size, init_kernel_size)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        
        
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # print(time) #
        # print(time.shape)#[40]
        # print(t.shape) #[40, 128]
        

        # classifier free guidance
        # torch.Size([16, 768]) torch.Size([1, 768]) torch.Size([16, 128])
        # print(cond.shape, self.null_cond_emb.shape, t.shape) #[16, 768]  [16, 128]
        # print(torch.cat((t, cond), dim = -1).shape) #[16, 896]
        # assert 2>111
        if self.has_cond:
            batch, device = x.shape[0], x.device
            # print(null_cond_prob) #=0
            mask = prob_mask_like((batch,), null_cond_prob, device = device) ##当 null_cond_prob = 0 时，会返回一个全为 False 的布尔张量。
            #torch.where(condition, x, y) 是一个条件选择函数，用于根据 condition 张量的布尔值选择 x 或 `y。
            #如果 condition 对应位置的值为 True，选择 x 的值。 如果 condition 对应位置的值为 False，选择 y 的值。
            # print(self.null_cond_emb.shape, self.null_cond_emb) #torch.Size([1, 768])
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1) #[40, 896]

        
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            # print(x.shape, t.shape, block1)
            x = block1(x, t)
            # print(x.shape, t.shape, block2)
            x = block2(x, t)
            # print(x.shape, t.shape)
            x = spatial_attn(x)
            # print(x.shape, t.shape)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            # print(x.shape, t.shape)
            h.append(x)
            x = downsample(x)
            # print(x.shape, t.shape)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        out_dim = 20
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.out_dim = out_dim

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        # print(x_t.shape, t.shape, noise.shape)
        # # torch.Size([16, 21, 7, 16, 16]) torch.Size([16]) torch.Size([16, 20, 7, 16, 16])
        # assert 3>333
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, src_cond=None, cond = None, cond_scale = 1., src_cond_scale = 1.0):
        # print(x.shape, t.shape, src_cond.shape)
        # torch.Size([16, 7, 20, 16, 16]) torch.Size([16]) torch.Size([1, 2, 20, 16, 16])
        inputX = torch.cat((x, src_cond), dim = 1) if exists(src_cond) else x
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(inputX, t, cond = cond, cond_scale = cond_scale, src_cond_scale = src_cond_scale))


        # x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(inputX, t, cond = cond, cond_scale = cond_scale)).clamp(-1, 1)

        # if clip_denoised:
        #     s = 1.
        #     if self.use_dynamic_thres:
        #         s = torch.quantile(
        #             rearrange(x_recon, 'b ... -> b (...)').abs(),
        #             self.dynamic_thres_percentile,
        #             dim = -1
        #         )

        #         s.clamp_(min = 1.)
        #         s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

        #     # clip by threshold, depending on whether static or dynamic
        #     x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, src_cond = None, cond = None, cond_scale = 1., src_cond_scale = 1.0, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, src_cond=src_cond, cond = cond, cond_scale = cond_scale, src_cond_scale = src_cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # print(nonzero_mask) #torch.Size([16, 1, 1, 1, 1]) nonzero_mask's value is 1 when t != 0, 0 when t == 0
        # print(model_log_variance.max(), model_log_variance.min())
        # assert 3>333
        # ipdb.set_trace()
        res = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        # print("p_sample_res~~~~~~~~~~~~~~~", res.shape, torch.max(res), torch.min(res))  #[16, 1, 7, 64, 64]
        # assert 4>444
        return res

    @torch.inference_mode()
    def p_sample_loop(self, shape, src_cond=None, cond = None, cond_scale = 1., src_cond_scale=1.0):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # if i % 10 == 0:
            #     ipdb.set_trace()
                # pudb.set_trace()
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), src_cond=src_cond, cond = cond, cond_scale = cond_scale, src_cond_scale = src_cond_scale, clip_denoised=False)

        # pudb.set_trace()
        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, src_cond = None, cond = None, cond_scale = 1., src_cond_scale=1.0, batch_size = 16):
        device = next(self.denoise_fn.parameters()).device
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, self.out_dim, num_frames, image_size, image_size), src_cond=src_cond, cond = cond, cond_scale = cond_scale, src_cond_scale=src_cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, src_cond=None, cond = None, noise = None, null_cond_prob=0.,  null_src_cond_prob=0., **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)

        # x_noisy = torch.cat((x_noisy, src_cond), dim = 1) if exists(src_cond) else x_noisy
        # null_cond_prob=0.5;   null_src_cond_prob=0.5
        # print(null_cond_prob, null_cond_prob)
        # assert 3>333
        x_recon = self.denoise_fn(x_noisy, t, src_cond=src_cond, cond = cond, \
                                  null_cond_prob=null_cond_prob, null_src_cond_prob=null_src_cond_prob, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, src_cond=None, text_cond = None, null_cond_prob=0., null_src_cond_prob=0., *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # x = normalize_img(x)
        return self.p_losses(x, t, src_cond=src_cond, cond=text_cond, \
                             null_cond_prob=null_cond_prob, null_src_cond_prob=null_src_cond_prob,*args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    #tensor: [7, 20, 80, 80]  for way2: frames=20
    tensor = tensor[:3]
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t
    return t * 2 - 1

def unnormalize_img(t):
    return t
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask,
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                


                batches = num_to_groups(num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')





# trainer class
class Trainer_with_GDN(object):
    def __init__(
        self,
        diffusion_model,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))


    def load_model2(self, Config, **kwargs):
        snapshot_path = Config["general"]["snapshot_path"]
        
        startIdx = 100
        DiFuS_mode_path = os.path.join(snapshot_path,   f'model-{startIdx}.pt')

        while not os.path.exists(DiFuS_mode_path) and startIdx>=0:
            startIdx -= 1
            DiFuS_mode_path = os.path.join(snapshot_path,   f'model-{startIdx}.pt')

        data = torch.load(DiFuS_mode_path)
        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    

    def load_model(self, path, **kwargs):
        data = torch.load(path)
        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])




    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        data,
        src_cond = None,
        text_cond = None, 
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
        Config = None,
        epoch = 0,
        save_model_flag = False,
        null_cond_prob=0.0, null_src_cond_prob=0.0
    ):
        assert callable(log_fn)
        # print(null_cond_prob, null_src_cond_prob)
        # assert 3>444
        self.model.train()
        # print("src_cond~~~~",src_cond.shape) #src_cond~~~~ torch.Size([25, 2, 20, 16, 16])
        with autocast(enabled = self.amp):
            loss = self.model(
                data,
                src_cond = src_cond,
                text_cond = text_cond,
                prob_focus_present = prob_focus_present,
                focus_present_mask = focus_present_mask,
                null_cond_prob=null_cond_prob, null_src_cond_prob=null_src_cond_prob
            )

            self.scaler.scale(loss / self.gradient_accumulate_every).backward()

        log = {'loss': loss.item()}

        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        if self.step % self.update_ema_every == 0:
            self.step_ema()

       
        ##  最终会生成16个 （4*4）
        Flag = True
        if epoch % self.save_and_sample_every == 0 and save_model_flag and Flag:
            milestone = epoch // self.save_and_sample_every
            num_samples = self.num_sample_rows ** 2
                        # print(num_samples, self.batch_size) #16 32
            batches = num_to_groups(num_samples, self.batch_size)
            # print(batches)
            if text_cond is not None:
                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n, cond = text_cond[:n] , src_cond=src_cond[:n]), batches))
            else:
                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n, src_cond=src_cond[:n]), batches))
            all_videos_list = torch.cat(all_videos_list, dim = 0)
            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))


            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
            if Config["general"]["DifuS_version"] == "way1": #[20,7,16,16] frames=7
                pass
            elif Config["general"]["DifuS_version"] == "way2":#[7,20,16,16] frames=20
                one_gif = one_gif.permute(0,2,1,3,4) #-> [7, 20, 7, 16, 16]  frames=7

            


            video_path = str(self.results_folder / str(f'{milestone}.gif'))
            video_tensor_to_gif(one_gif, video_path)
            log = {**log, 'sample': video_path}
            self.save(milestone)
            
        # print("step:     ", self.step)

        self.step += 1
        return loss
    

    def predict(
        self,
        src_cond = None,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
        Config = None,
        Index = None
    ):
        assert callable(log_fn)
        self.model.eval()
        # print("src_cond~~~~",src_cond.shape) #src_cond~~~~ torch.Size([25, 2, 20, 16, 16])

        ##  最终会生成16个 （4*4）
        num_samples = self.num_sample_rows ** 2
        # print(num_samples, self.batch_size) #16 32
        batches = num_to_groups(num_samples, self.batch_size)
        # print(batches)

        # src_cond: [1, 2, 20, 16, 16] or [1, 1, 20, 16, 16]
        # src_cond = src_cond.reapeat(16,1,1,1,1)

        all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n, src_cond=src_cond[:16]), batches))
        # print("all_videos_list~~~~",len(all_videos_list), all_videos_list[0].shape)


        # print("all_videos_list~~~~", all_videos_list.shape) #torch.Size([16, 7, 20, 20, 20])█-> [16, 20, 7, 20, 20] frames=7
        
        if Config["general"]["DifuS_version"] == "way1": #[16,20,7,16,16] frames=7
            pass
        elif Config["general"]["DifuS_version"] == "way2":#[16,7,20,16,16] frames=20 -> [16, 20, 7, 16, 16]  frames=7
            all_videos_list = [x.permute(0,2,1,3,4) for x in all_videos_list]

        # all_videos_list~~~~ 1 torch.Size([16, 7, 20, 16, 16])  for way2: frames=20 -> [16, 20, 7, 16, 16] frames=7
        all_videos_list = torch.cat(all_videos_list, dim = 0)
        all_videos_list_lz = all_videos_list  #[b, 20, 7, 16, 16] frames=7

        all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
        
        # print("one_gif~~~~", one_gif.shape) #[7, 20, 80, 80]  for way2: frames=20
        one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
        video_path = str(self.results_folder / str(f'{Index}.gif'))
        video_tensor_to_gif(one_gif, video_path)
        return all_videos_list_lz


        
    def test(
        self,
        data,
        src_cond = None,
        text_cond = None,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)
        self.model.eval()
        with autocast(enabled = self.amp):
            loss = self.model(
                data,
                src_cond = src_cond,
                text_cond = text_cond,
                prob_focus_present = prob_focus_present,
                focus_present_mask = focus_present_mask
            )
        return loss
