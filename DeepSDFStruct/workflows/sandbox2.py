import torch
import logging
import socket
import math
import tqdm
import time

import DeepSDFStruct.deep_sdf
import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.networks.deep_sdf_discriminator import ConvDiscriminator
from DeepSDFStruct.deep_sdf.GAN_helpers.gan_sampling import *
from DeepSDFStruct.sdf_primitives import CrossMsSDF
from DeepSDFStruct.deep_sdf.GAN_helpers.Hinge_GAN_loss import *
from DeepSDFStruct.deep_sdf.plotting import *

device = torch.device("cpu")
experiment_directory = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiments/gan_test_experiment16"

#load experiment specs
specs = ws.load_experiment_specifications(experiment_directory)


real_sdf = CrossMsSDF(0.0)
decoder = ws.init_decoder(specs, device, data_parallel = False).to(device) #data_paralell must be set to true if muliple compute devices are active

sampler = ConvGAN_SDF_Sampler(real_sdf, decoder, 2, 2, specs["SdfParameterBounds"], device, add_latent= True)

samples = sampler.fetch_samples()


"""
print("REAL")
print(samples[0])
print(samples[0].shape)
print()
print("FAKE")
print(samples[1])
print(samples[1].shape)
print("Latent Vecs:")
print(samples[2]) #type: ignore
"""