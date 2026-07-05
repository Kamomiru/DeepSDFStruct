"""
SDF Discriminator for GAN Training
"""

import torch.nn as nn
import torch

sdf_clamp_val = 1.9 #if SDFs are somewhat symetric around 0,0,0 the max SDF value should never be higher than sqrt(3) = 1.7321. Hence we choose a value slightly above that so all extreme and unrealistic values sill stand out

class ConvDiscriminator(nn.Module):
    def __init__(self, clampSDF):
        super(ConvDiscriminator, self).__init__()
        self.clampSDF = clampSDF


        #how will the current latvec be taken into account here? 2 input chanels in conv3d?
        self.net = nn.Sequential(
            #Spatial: 32³ -> 16³
            nn.Conv3d(1, 32, 4, stride = 2, padding = 1), #1 input feature (SDF Value) -> 32 output features | Spacial dimension gets halved since stride = 2. 32³ -> 16³
            nn.LeakyReLU(0.2),

            #Spatial: 16³ -> 8³
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm3d(64), #weight normalization | eventually ue nn.BatchNorm3d?
            nn.LeakyReLU(0.2),

            #Spatial: 8³ -> 4³
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2),

            #Spatial: 4³ -> 2³
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2),

            #(Batches, Feature Chanels = 256, 2, 2, 2) -> (Batches, Feature Chanels = 256, 1, 1, 1)
            nn.AdaptiveAvgPool3d(1) #pool all features into one singular output
        )

        self.lin = nn.Linear(256, 1)

    def forward(self, x):
        if self.clampSDF == True: #clamping already implemented in training.py
            x = torch.clamp(x, -sdf_clamp_val, sdf_clamp_val)
        x = self.net(x)
        x = x.flatten(1) #Reduce (Batches, Feature Chanels = 256, 1, 1, 1) -> (Batches, Feature Chanels = 256, 1)

        #implement latent vector injection at Discriminator?
        return self.lin(x)
        

