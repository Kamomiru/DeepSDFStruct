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

#----Open Variables----
num_embeddings = 10
embedding_dim = 1
n_nodes = 32

code_bound = 1.0
CodeInitStdDev = 1.0

lr_decoder = 0.01
lr_discriminator = 0.005
lr_latent = 0.01

samples_per_batch = 8
batch_per_epoch = 4
max_num_epochs = 5


if samples_per_batch * batch_per_epoch % 2 != 0:
    raise RuntimeError("samples_per_batch * batch_per_epoch must be divisible by 2 to ensure equal amounts of real and fake inputs for discriminator!")
num_real_samples = batch_per_epoch*samples_per_batch // 2



#start logger
logger = logging.getLogger(DeepSDFStruct.__name__)

def train_deeep_sdf(
    experiment_directory, data_source, continue_from=None, batch_split=1, device=None
):
    #----Compute Device Checking----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        device_name = torch.cuda.get_device_name()
    elif device == "cpu":
        device_name = "cpu"
    else:
        raise RuntimeError("Device must be either cpu or cuda")
    
    #load experiment specs
    specs = ws.load_experiment_specifications(experiment_directory)
    logger.info(f"Reading experiment configuration from {experiment_directory}")
    logger.info("Experiment description: \n" + specs["Description"])

    logger.debug(specs["NetworkSpecs"])

    #initialize decoder
    decoder = ws.init_decoder(specs, device, data_parallel = False) #data_paralell must be set to true if muliple compute devices are active
    #initialize discriminator
    discriminator = ConvDiscriminator(clampSDF = True)

    if device == "cuda":
        device_name = torch.cuda.get_device_name()
    elif device == "cpu":
        device_name = "cpu"
    else:
        raise RuntimeError("Device must be either cpu or cuda")
    
    host_name = socket.gethostname()
    logger.info(f"training on {host_name} with {device_name}")
    
    #initialize latent vectors
    lat_vecs = torch.nn.Embedding(
        num_embeddings, embedding_dim, max_norm=code_bound, device=device
    )
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, CodeInitStdDev / math.sqrt(embedding_dim))

    #initialize optimizers
    optimizer_dec = torch.optim.adam({"params": decoder.parameters(),
                                      "lr": lr_decoder})
    optimizer_disc = torch.optim.adam({"params": discriminator.parameters(),
                                      "lr": lr_discriminator})
    optimizer_lat = torch.optim.adam({"params": lat_vecs.parameters(),
                                      "lr": lr_latent})
    
    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    epoch = 1

    meshgrid = create_meshgrid(n_nodes = 32)
    
    real_sdf = real_sdf = CrossMsSDF(0.4)
    ConvGAN_SDF_Sampler(real_sdf, deep_sdf, 2, 8, (0,1.0))

    start_train = time.time()
    pbar = tqdm.trange(epoch, max_num_epochs + 1, desc="Training", smoothing=0)
    for epoch in pbar:
        epoch_error = 0.0
        start = time.time()

        #IMPLEMENT ADJUSTABLE LEARNING RATE

        for batch in range(batch_per_epoch):

            #TBC.

