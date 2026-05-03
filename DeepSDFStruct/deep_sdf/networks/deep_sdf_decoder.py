"""
Standard DeepSDF Decoder Network
================================

This module implements the standard DeepSDF decoder architecture from
Park et al. (2019). The network takes a latent code and 3D coordinates
as input and outputs a signed distance value.

Architecture
------------
The decoder is a multi-layer perceptron (MLP) with:
- Concatenation of latent code and spatial coordinates at input
- Optional skip connections for latent code injection
- Optional coordinate injection at multiple layers
- Batch normalization or weight normalization
- ReLU activations (or tanh for final layer)
- Dropout for regularization

The network learns to map from a latent space to implicit SDF representations,
enabling compact encoding of complex 3D geometries.
"""

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size, 
        dims, #List of our hidden Layer Dimensions
        geom_dimension, 
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(), #sets wich layers should be subject to weight paraemtrization
        latent_in=(),
        weight_norm=False, #activates network weight parametrization
        xyz_in_all=None, #set True if xyz data should be injected in each layer
        use_tanh=False,
        latent_dropout=False,
    ):
        super(DeepSDFDecoder, self).__init__()

        def make_sequence():
            return []


        #--------------------Class Setup--------------------
        dims = [latent_size + geom_dimension] + dims + [1] #List of Layer Dimensions WITH input and output layer
        #e.g. [5, 128, 128, 128, 128, 1]

        self.num_layers = len(dims)
        self.geom_dimension = geom_dimension
        self.norm_layers = norm_layers
        self.latent_in = latent_in #?defines the layer where our latent vector gets injected?
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        #--------------------Layer Setup--------------------
        for layer in range(0, self.num_layers - 1):
            #----Handling latent injection----
            if layer + 1 in latent_in: #if the next layer is our latent vector input
                out_dim = dims[layer + 1] - dims[0] #make our output dim smaller so we can later inject our latent vector + xyz
            else:
                out_dim = dims[layer + 1] #simply sets layers output to next layers input 
                if self.xyz_in_all and layer != self.num_layers - 2: #injection of xyz in all layers
                    out_dim -= geom_dimension

            #----Weight normalization----
            #also decouples weight(vector) magnitude from direction -> easier optimization
            if weight_norm and layer in self.norm_layers:
                setattr(self, "lin" + str(layer),nn.utils.parametrizations.weight_norm(nn.Linear(dims[layer], out_dim))) #Creates a normalized linear layer. The weights(vector) direction and magnitude can now be learned independent from each other
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim)) #Linear layer without weigth parametrization


            #----Activation normalization layers----
            if ((not weight_norm) 
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3), or N x (L+geom_dimension)
    def forward(self, input):
        xyz = input[:, -self.geom_dimension :] #Extracts only our positional data XYZ

        #----Applying Latent Vector Dropout----
        if input.shape[1] > self.geom_dimension and self.latent_dropout:
            latent_vecs = input[:, : -self.geom_dimension] #Gets the rest of our input (without xyz) -> Latent Vectors
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training) #training= true/false sets whether dropout is applied or not
            x = torch.cat([latent_vecs, xyz], 1) #Recombining Positional Data with latent vectors that went through dropout func
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            #----Use tanh on last layer----
            if layer == self.num_layers - 2 and self.use_tanh: 
                x = self.tanh(x)
            #----Layer Normalization, Activation using ReLu and dropout for all but the last layer----
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x) #layer activation Normalization
                x = self.relu(x) #activation function
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training) #dropout

        return x
