import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break
    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss


class Evaluator:
    def __init__(
        self, eval_loader, logger, evalmode="fast", debug=False):
        self.eval_loader = eval_loader
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm):
        summaries = collections.defaultdict(float)
        # for key order
        summaries["val_accuracy"] = 0.0
        summaries["val_loss"] = 0.0

        acc, loss = accuracy_from_loader(algorithm, self.eval_loader, None, debug=self.debug)
        summaries["val_accuracy"] = acc
        summaries["val_loss"] = loss
        return summaries
