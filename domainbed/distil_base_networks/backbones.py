import timm
import torch
import torch.nn as nn
import torchvision.models
import clip
from .convnext import convnext_base, convnext_small
from .swin_transformer import get_small_win, get_base_win
from .vit import vit_base_patch16

class DINO_Encoder(nn.Module):
    def __init__(self, dino_model, use_avgpool=False):
        super(DINO_Encoder, self).__init__()
        self.feature_model = dino_model
        self.use_avgpool = use_avgpool

    def forward(self, x, **kwargs):
        features = self.feature_model(x)
        return features.float()

class CLIP_Image_Encoder(nn.Module):
    def __init__(self, clip_model):
        super(CLIP_Image_Encoder, self).__init__()
        self.clip_model = clip_model
        self.output_dim = clip_model.output_dim

    def forward(self, x, **kwargs):
        return self.clip_model(x)

class Null_Head_Backbone(nn.Module):
    def __init__(self, model, output_dim, freeze_bn=False):
        super(Null_Head_Backbone, self).__init__()
        self.model = model
        self._freeze_bn = freeze_bn
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self._freeze_bn is False:
            return
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def get_backbone(name, pretrained=True, freeze_bn=True, **kwargs):
    if name == "CLIP_img_encoder_VIT-B":
        model_name = "ViT-B/16"
        return CLIP_Image_Encoder(clip.load(model_name, device="cuda")[0].visual.float())
    elif name == "CLIP_img_encoder_VIT-L":
        model_name = "ViT-L/14"
        return CLIP_Image_Encoder(clip.load(model_name, device="cuda")[0].visual.float())
    elif name == "DINO_V2_VIT-B":
        dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        return DINO_Encoder(dinov2_vitb14_reg)
    elif name == "DINO_V2_VIT-S":
        dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        return DINO_Encoder(dinov2_vits14_reg)
    elif name == "convnext-B":
        model = convnext_base(pretrained=pretrained, in_22k=True)
        model.head = nn.Identity()
        output_dim = model.head.weight.data.shape[0]
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    elif name == "convnext-S":
        model = convnext_small(pretrained=pretrained, in_22k=True)
        output_dim = model.head.weight.data.shape[0]
        model.head = nn.Identity()
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    elif name == "swin_transformer-S":
        model = get_small_win(pretrained=pretrained, in_22k=True)
        model.head = nn.Identity()
        output_dim = model.num_features
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    elif name == "swin_transformer-B":
        model = get_base_win(pretrained=pretrained, in_22k=True)
        model.head = nn.Identity()
        output_dim = model.num_features
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    elif name == "mae-B":
        model = vit_base_patch16(pretrained=pretrained)
        model.head = nn.Identity()
        output_dim = model.embed_dim
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    elif name == "ResNet-50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        output_dim = 2048
        return Null_Head_Backbone(model, output_dim, freeze_bn)
    else:
        raise

