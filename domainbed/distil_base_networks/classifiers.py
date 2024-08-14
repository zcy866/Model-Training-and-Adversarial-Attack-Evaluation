import timm
import torch
import torch.nn as nn
import torchvision.models
import clip
import torch.nn.functional as F

def get_template(prompt_style):
    # type 1 - {class}
    if prompt_style == 1:
        print("Prompt Template: \{class\}")
        templates = ["{class_name}"]

    # type 2 - a photo of a {class}
    elif prompt_style == 2:
        print("Prompt Template: a photo of a \{class\}")
        templates = ["a photo of a {class_name}"]

    # type 3 - a {dom} photo of a {class}
    elif prompt_style == 3:
        print("Prompt Template: a \{dom\} photo of a \{class\}")
        templates = [
            "an art of a {class_name}",
            "a clipart of a {class_name}",
            "a photo of a {class_name} product",
            "a photo of a {class_name}",
        ]

    # type 4 - a {dom} photo of a {small / big} {class}
    elif prompt_style == 4:
        print("Prompt Template: a \{dom\} photo of a \{small/big\} \{class\}")
        templates = [
            "an art photo of a {size} {{class_name}}",
            "a clipart photo of a {size} {{class_name}}",
            "a product photo of a {size} {{class_name}}",
            "a real photo of a {size} {{class_name}}",
        ]
    else:
        raise
    return templates


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.input_shape = input_size
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        inputs = F.normalize(inputs, dim=-1)
        return super().forward(inputs)


def get_zeroshot_classifier(clip_model, classnames, templates):
    logit_scale = clip_model.logit_scale

    print("Getting zeroshot weights.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [
                template.format(class_name=classname) for template in templates
            ]

            # Embeddings for each class
            texts = clip.tokenize(texts).cuda()  # tokenize
            embeddings = clip_model.encode_text(
                texts
            )  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)

        # Computing zero-shot weights
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(
        normalize=True, weights=zeroshot_weights
    )
    return classification_head

class Null_Classifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Null_Classifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def forward(self, x):
        bs = x.shape[0]
        return torch.zeros([bs, self.num_classes]).to(x.device)

class Norm_classifier(nn.Module):
    def __init__(self, input_shape, num_classes, temp=1):
        super(Norm_classifier, self).__init__()
        self.input_shape = input_shape
        self.layer = nn.Linear(input_shape, num_classes)
        self.temp = temp

    def forward(self, x):
        norm_x = F.normalize(x, dim=-1)
        return self.layer(norm_x) / self.temp

def get_classifier(name, class_names):
    if name == "CLIP_img_encoder_VIT-B":
        model_name = "ViT-B/16"
        clip_model = clip.load(model_name, device="cuda")[0]

        templates = get_template(2)
        clip_cls = get_zeroshot_classifier(
            clip_model, class_names, templates
        )
        return clip_cls
    elif name == "CLIP_txt_encoder_VIT-B":
        model_name = "ViT-B/16"
        clip_model = clip.load(model_name, device="cuda")[0]

        templates = get_template(2)
        clip_cls = get_zeroshot_classifier(
            clip_model, class_names, templates
        )
        return clip_cls
    elif name == "DINO_V2_VIT-B":
        input_shape = 768
        num_classes = len(class_names)
        return Null_Classifier(input_shape, num_classes)