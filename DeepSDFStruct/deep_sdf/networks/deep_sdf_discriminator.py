"""
SDF Discriminator for GAN Training
"""

import torch.nn as nn
import torch
import math

sdf_clamp_val = 1.9 #if SDFs are somewhat symetric around 0,0,0 the max SDF value should never be higher than sqrt(3) = 1.7321. Hence we choose a value slightly above that so all extreme and unrealistic values sill stand out

class ConvDiscriminator(nn.Module):
    def __init__(self, n_nodes, spectral_reg, latent_dim, use_latent_conditioning = False):
        super(ConvDiscriminator, self).__init__()

        self.HingeGAN = True #to be implemented if standard non-saturating GAN Loss should be used -> sigmoid function is needed.
        self.spectral_reg = spectral_reg

        self.use_latent_conditioning = use_latent_conditioning
        self.latent_dim = latent_dim
        self.latent_feature_dim = 64

        if n_nodes not in [4, 8, 16, 32, 64]:
            raise RuntimeError("n_nodes must be 4, 8, 16, 32 or 64!")
        self.n_nodes = n_nodes
        self.layers = int(math.log2(self.n_nodes/2))

        layers = []

        in_channels = 1
        out_channels = 32
        kernel_size = 4
        stride = 2
        padding = 1

        #--------Spatial Features--------
        for layer in range(1, self.layers + 1):

            if spectral_reg:
                layers.append(nn.utils.spectral_norm(
                    nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)))
                
            else:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))


            if layer != 1:
                layers.append(nn.InstanceNorm3d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2))

            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.AdaptiveAvgPool3d(1))

        self.net = nn.Sequential(*layers) # * is an unpacking operator

        # Number of features produced by the Conv3D network
        self.sdf_feature_dim = in_channels

        #--------Small network for Latent Vector handling--------
        self.latent_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2))
        

        if self.use_latent_conditioning == False:
            self.latent_feature_dim = 0
        #combine Spatial and Latent through last linear layer
        self.lin = nn.Linear( self.sdf_feature_dim + self.latent_feature_dim, 1 )

        self.sigm = nn.Sigmoid()

    def forward(self, x, latent): 

        # Process SDF Volume
        x = self.net(x) # [B, C, 1, 1, 1] -> [B, C]
        x = x.flatten(1)


        if self.use_latent_conditioning:
            # Process Latent Vector
            latent = self.latent_net(latent)

            # Combine SDF representation and Latent representation
            x = torch.cat([x, latent], dim=1)
        else:
            x = self.lin(x) # single output

        if self.HingeGAN == False:
            x = self.sigm(x)

        return x