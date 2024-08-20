import torch
from torch.optim.lr_scheduler import LambdaLR
import math

def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)

def get_cosine_schedule(optimizer,
                                num_training_steps,
                                num_wait_steps=0,
                                num_cycles=0.5,
                                last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_wait_steps:
            return float(current_step) / float(max(1, num_wait_steps))

        progress = float(current_step - num_wait_steps) / \
            float(max(1, num_training_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def build_scheduler(name, **kwargs):
    if name == "cosine_schedule":
        return get_cosine_schedule(**kwargs)
    if name == "const_scheduler":
        def lr_lambda(current_step):
            return 1
        return LambdaLR(kwargs["optimizer"], lr_lambda, -1)
