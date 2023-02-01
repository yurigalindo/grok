import torch
import torch.nn as nn

import random
import numpy as np

def parity(n, k, n_samples, seed=42):
    'Data generation'

    random.seed(seed)
    samples = torch.Tensor([[random.choice([-1, 1]) for j in range(n)] for i in range(n_samples)])
    # targets = torch.prod(input[:, n//2:n//2+k], dim=1) # parity hidden in the middle
    targets = torch.prod(samples[:, :k], dim=1) # parity hidden in first k bits

    return samples, targets

def mean_and_std_across_seeds(list_stats):
    array_stats = np.array(list_stats)
    mean = np.average(array_stats, axis=0)
    std =  np.std(array_stats, axis=0)

    return mean, std

def acc_calc(dataloader, model, mask_idx=None, device='cuda'):
    model.eval()

    acc, total = 0, 0
    for id, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if (mask_idx is not None):
            # create mask
            idx = torch.LongTensor([mask_idx for _ in range(len(y_batch))])
            mask = torch.zeros(len(y_batch), 1000)
            mask.scatter_(1, idx, 1.)

            pred = model.masked_forward(x_batch, mask.to(device))
        else:
            pred = model(x_batch)
        acc += (torch.sign(torch.squeeze(pred)) == y_batch).sum().item()
        total += x_batch.shape[0]
    
    return acc / total


def loss_calc(dataloader, model, loss_fn, device='cuda'):
    model.eval()

    loss, total = 0, 0
    for id, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)
        loss += loss_fn(pred, y_batch).sum().item()
        total += x_batch.shape[0]

    return loss / total

def circuit_discovery_linear(epoch, saved_model, norms, dataloader, device='cuda'):
    # Calculate least number of neurons that recovers original (train set) performance with linear search

    values = np.array(norms['feats'][epoch]).argsort()
    for k in range(1, 1000):
        idx = values[-k:]
        masked_acc = acc_calc(dataloader, saved_model, idx, device=device)
        full_acc = acc_calc(dataloader, saved_model, device=device)

        if (masked_acc == full_acc):
            return k, idx
    
    return float('inf'), None # mistake
    
    

def circuit_discovery_binary(epoch, saved_model, norms, dataloader, device='cuda'):
    # Calculate least number of neurons that recovers original (train set) performance with binary search (assuming that it increases monotonically)
    
    left, right = 1, 1000
    prev_k = -1
    min_k, min_idx = float('inf'), None

    values = np.array(norms['feats'][epoch]).argsort()
    while left < right:
        k = (left + right) // 2
        if (prev_k == k):
            break

        idx = values[-k:]
        masked_acc = acc_calc(dataloader, saved_model, idx, device=device)
        full_acc = acc_calc(dataloader, saved_model, device=device)

        if (masked_acc == full_acc) and (k < min_k):
            min_k = k
            min_idx = idx
        if (masked_acc < full_acc):
            left = k
        else:
            right = k + 1

        prev_k = k
    
    return min_k, min_idx

class FF1(torch.nn.Module):
    def __init__(self, input_dim=40, width=1000):
        super(FF1, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, width)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    def masked_forward(self, x, mask):
        x = self.linear1(x)
        x = self.activation(x)
        x = x * mask
        x = self.linear2(x)
        return x

class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(torch.squeeze(output), torch.squeeze(target))
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss
