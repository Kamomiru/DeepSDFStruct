"""
SDF Classifier
Classifies the decoders output SDF and attempts to predict the decoders input parameters.
"""

import torch.nn as nn
import torch
import math

class ConvClassifier(nn.Module):
    def __init__(self, n_nodes, spectral_reg, conditioning_dim):
        super(ConvClassifier, self).__init__()
        self.spectral_reg = spectral_reg
        self.conditioning_dim = conditioning_dim

        if n_nodes not in [4, 8, 16, 32, 64]:
            raise RuntimeError("n_nodes must be 4, 8, 16, 32 or 64!")
        self.n_nodes = n_nodes
        self.layers = int(math.log2(self.n_nodes/2))

        layers = []
        
        in_chanels = 1
        out_channels = 32
        kernel_size = 4
        stride = 2
        padding = 1

        for layer in range(1, self.layers + 1):
        
            if spectral_reg:
                layers.append(nn.utils.spectral_norm(
                    nn.Conv3d(in_chanels, out_channels, kernel_size, stride, padding)))
                
            else:
                layers.append(nn.Conv3d(in_chanels, out_channels, kernel_size, stride, padding))


            if layer != 1:
                layers.append(nn.InstanceNorm3d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2))

            in_chanels = out_channels
            out_channels *= 2
        
        layers.append(nn.AdaptiveAvgPool3d(1))

        self.net = nn.Sequential(*layers)

        #implement larger network after convolutional layers?
        self.lin = nn.Linear(in_chanels, conditioning_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)

        x = self.lin(x)

        return x.squeeze(-1) #remove last singular dimension