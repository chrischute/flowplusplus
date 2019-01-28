import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
        random_init (bool): Initialize with a random orthogonal matrix.
            Otherwise initialize with noisy identity.
    """
    def __init__(self, num_channels, random_init=False):
        super(InvConv, self).__init__()
        self.num_channels = 2 * num_channels

        if random_init:
            # Initialize with a random orthogonal matrix
            w_init = np.random.randn(self.num_channels, self.num_channels)
            w_init = np.linalg.qr(w_init)[0]
        else:
            # Initialize as identity permutation with some noise
            w_init = np.eye(self.num_channels, self.num_channels) \
                     + 1e-3 * np.random.randn(self.num_channels, self.num_channels)
        self.weight = nn.Parameter(torch.from_numpy(w_init.astype(np.float32)))

    def forward(self, x, sldj, reverse=False):
        x = torch.cat(x, dim=1)

        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        x = F.conv2d(x, weight)
        x = x.chunk(2, dim=1)

        return x, sldj
