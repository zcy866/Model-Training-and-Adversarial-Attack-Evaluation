import copy
import math

import timm
import torch
import torch.nn as nn
import torchvision.models
import clip
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sconf import Config
import warnings
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def norm_cross_atten(q, k, v):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    weight = (q @ k.transpose(-1, -2))*math.sqrt(q.shape[-1])
    weight = weight.softmax(dim=-1)
    out = weight @ v.transpose(0, 1)
    return out.transpose(0, 1)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, start_layer=0, end_layer=-1, mid_record_layers=[], attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.start_layer = start_layer
        if end_layer == -1:
            self.end_layer = layers
        self.mid_record_layers = mid_record_layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        mid_list = []
        for i in range(self.start_layer, self.end_layer):
            x = self.resblocks[i](x)
            if i in self.mid_record_layers:
                mid_list.append(x)
        return x, mid_list


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, 
                 start_layer: int, end_layer: int, mid_record_layers: list, is_mid_cls_only: bool = False):
        super().__init__()
        self.start_layer = start_layer
        self.is_mid_cls_only = is_mid_cls_only
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, start_layer=start_layer, end_layer=end_layer, mid_record_layers=mid_record_layers)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def dual_proj(self, x):
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def forward(self, x: torch.Tensor):
        if self.start_layer == 0:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND    

        x, mid_list = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        if self.is_mid_cls_only:
            for i in range(len(mid_list)):
                mid_list[i] = mid_list[i].permute(1, 0, 2)
                mid_list[i] = self.ln_post(mid_list[i][:, 0, :])
                if self.proj is not None:
                    mid_list[i] = mid_list[i] @ self.proj

        return x, mid_list

class singleTaskVisual(nn.Module):
    def __init__(self, clip_visual: nn.Module, start_layer: int, end_layer: int, mid_record_layers: list, is_mid_cls_only: bool):
        super(singleTaskVisual, self).__init__()
        self.input_resolution = clip_visual.input_resolution
        self.patch_size = clip_visual.conv1.kernel_size[0]
        self.width = clip_visual.transformer.width
        self.layers = clip_visual.transformer.layers
        self.heads = clip_visual.transformer.resblocks[0].attn.num_heads
        self.output_dim = clip_visual.output_dim
        self.insert_layer = start_layer

        self.model = VisionTransformer(self.input_resolution, self.patch_size, self.width, self.layers, self.heads, self.output_dim, start_layer, end_layer, mid_record_layers, is_mid_cls_only)
        model_params = clip_visual.state_dict()
        self.model.load_state_dict(model_params, strict=True)

    def forward(self, x: torch.Tensor):
        x, mid_list = self.model(x)
        return x, mid_list

class CombVisualClip(nn.Module):
    def __init__(self, model_name: str, insert_layer: int, is_multi_mid: bool, device: str = "cuda"):
        super(CombVisualClip, self).__init__()
        self.is_multi_mid = is_multi_mid

        base_model = clip.load(model_name, device=device)[0].visual.float()

        mid_dim = base_model.transformer.width
        out_dim = base_model.output_dim
        layer_lens = base_model.transformer.layers
        if self.is_multi_mid == layer_lens-1:
            warnings.warn("Note that the value of is_multi_mid becomes invalid because insert_layer is the last layer.", category=RuntimeWarning)
            is_multi_mid = False
            self.is_multi_mid = False
        self.input_adapter = nn.Linear(mid_dim, mid_dim)
        self.main_output_adapter = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

        sub_mid_record_layers = []
        self.output_adapters = [self.main_output_adapter]
        if is_multi_mid:
            self.mask_weight = nn.Parameter(torch.tensor(torch.rand(layer_lens-insert_layer)))
            sub_mid_record_layers = [insert_layer]
            for i in range(layer_lens-insert_layer-2): 
                sub_mid_record_layers.append(sub_mid_record_layers[-1]+1)

            for i in range(layer_lens-insert_layer-1):
                setattr(self, "output_adapter" + str(i),
                    nn.Sequential(nn.Linear(out_dim, out_dim),nn.Sigmoid()))
                self.output_adapters.append(getattr(self, "output_adapter" + str(i)))
        assert insert_layer >= 1
        self.main_model = singleTaskVisual(base_model, start_layer=0, end_layer=-1, mid_record_layers=[insert_layer-1], is_mid_cls_only=False)
        self.sub_model = singleTaskVisual(base_model, start_layer=insert_layer, end_layer=-1, mid_record_layers=sub_mid_record_layers, is_mid_cls_only=True)

    def forward(self, x:torch.Tensor):
        main_out, sub_out = self.main_model(x)
        sub_out = self.input_adapter(sub_out[0])
        self.sub_model.eval()
        with torch.no_grad():
            mask1, mask_list = self.sub_model(sub_out)
        if self.is_multi_mid:
            all_mask = [mask1]+mask_list
            weight = F.softmax(self.mask_weight, dim=-1)
            for i in range(len(all_mask)):
                all_mask[i] = self.output_adapters[i](all_mask[i]) * weight[i]
            final_mask = sum(all_mask)
        else:
            final_mask = mask1
        out = main_out #final_mask * main_out + 
        return out
    
    def frozen_paramaters(self, frozen_layers, active_layers):

        for name, param in self.main_model.named_parameters():
            for tname in frozen_layers:
                if name.find(tname) >= 0:
                    is_used = True
                    for ttname in active_layers:
                        if name.find(ttname) >= 0:
                            is_used = False
                            break
                    if is_used:
                        param.requires_grad = False
                        break

