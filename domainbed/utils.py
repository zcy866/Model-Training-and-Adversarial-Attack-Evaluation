import torch
import copy
import torch.nn as nn

class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model).cuda()
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self, model):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module

class ModelEMA(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=False):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.cuda()#to(device=device)

    def forward(self, x):
        return self.module(x)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.cuda()#to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.cuda()#to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


    def eval(self):
        self.module.eval()