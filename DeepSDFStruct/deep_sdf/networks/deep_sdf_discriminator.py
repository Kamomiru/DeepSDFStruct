"""
SDF Discriminator for GAN Training
"""

import torch.nn as nn
import torch
import math

sdf_clamp_val = 1.9 #if SDFs are somewhat symetric around 0,0,0 the max SDF value should never be higher than sqrt(3) = 1.7321. Hence we choose a value slightly above that so all extreme and unrealistic values sill stand out

class ConvDiscriminator(nn.Module):
    def __init__(self, n_nodes):
        super(ConvDiscriminator, self).__init__()
        self.HingeGAN = True #to be implemented if standard non-saturating GAN Loss should be used -> sigmoid function is needed.
        
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
            layers.append(nn.Conv3d(in_chanels, out_channels, kernel_size, stride, padding))


            if layer != 1:
                layers.append(nn.InstanceNorm3d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2))

            in_chanels = out_channels
            out_channels *= 2

        layers.append(nn.AdaptiveAvgPool3d(1))

        self.net = nn.Sequential(*layers)

        self.lin = nn.Linear(in_chanels, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1) #Reduce (Batches, Feature Chanels = 256, 1, 1, 1) -> (Batches, Feature Chanels = 256, 1)

        #implement latent vector injection at Discriminator?
        x = self.lin(x)

        if self.HingeGAN == False:
            x = self.sigm(x)
        return x
        

