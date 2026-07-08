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

#----Open Variables----



#start logger
logger = logging.getLogger(DeepSDFStruct.__name__)
#logger.setLevel(logging.DEBUG)

def train_deep_sdf_gan(
    experiment_directory, continue_from=None, batch_split=1, device=None
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
    GAN_architecture = specs["GANArchitecture"]

    if specs["n_nodes"] != 32:
        raise RuntimeError(f"n_nodes must be set to 32 since the discriminator is not yet implemented to handle arbitrary node numbers!\nn_nodes is set to {specs["n_nodes"]}")

    logger.debug(specs["NetworkSpecs"])

    if device == "cuda":
        device_name = torch.cuda.get_device_name()
    elif device == "cpu":
        device_name = "cpu"
    else:
        raise RuntimeError("Device must be either cpu or cuda")
    
    host_name = socket.gethostname()
    logger.info(f"training on {host_name} with {device_name}")

    #initialize decoder
    decoder = ws.init_decoder(specs, device, data_parallel = False).to(device) #data_paralell must be set to true if muliple compute devices are active
    #initialize discriminator
    discriminator = ConvDiscriminator(clampSDF = True).to(device)

    #initialize optimizers
    optimizer_dec = torch.optim.Adam(decoder.parameters(),
                                      lr = specs["InitialLearningRates"]["decoder"])
    optimizer_disc = torch.optim.Adam(discriminator.parameters(),
                                      lr = specs["InitialLearningRates"]["discriminator"])

    #Batch size Variables and checking
    samples_per_batch = specs["SamplesPerBatch"]
    batch_per_epoch = specs["BatchPerEpoch"]
    samples_per_epoch = samples_per_batch * batch_per_epoch

    if samples_per_epoch % 2 != 0:
            raise RuntimeError("samples_per_batch * batch_per_epoch must be divisible by 2 to ensure equal amounts of real and fake inputs for discriminator!")
    num_real_samples = batch_per_epoch*samples_per_batch // 2

    #Data Generation/Sampling
    real_sdf = real_sdf = CrossMsSDF(0.0)
    sampler = ConvGAN_SDF_Sampler(real_sdf, decoder, specs["n_nodes"], samples_per_batch, specs["SdfParameterBounds"], device)

    #Training stat logging
    loss_log_D = []
    loss_log_G = []
    lr_log_D = []
    lr_log_G = []
    #lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    avg_real_score_D_log = []
    avg_fake_score_D_log = []

    decoder.train()
    discriminator.train()

    epoch = 1
    start_train = time.time()
    pbar = tqdm.trange(epoch, specs["NumEpochs"] + 1, desc="Training", smoothing=0)
    for epoch in pbar:
        start = time.time()
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        total_real_score_D = 0.0
        total_fake_score_D = 0.0
        
   

        

        #IMPLEMENT ADJUSTABLE LEARNING RATE

        for batch in range(batch_per_epoch):
            optimizer_disc.zero_grad()
            optimizer_dec.zero_grad()

            real_batch, fake_batch = sampler.fetch_samples()

            #Train Discriminator
            real_scores = discriminator(real_batch)
            fake_scores_d = discriminator(fake_batch.detach()) #We need to detach the fake_batch so all our fake samples are treated as constant and our G gradients dont flow into our D gradient

            loss_D = Hinge_Loss_D(real_scores, fake_scores_d)
            loss_D.backward()
            optimizer_disc.step()

            #Train Generator
            fake_batch = sampler.fetch_samples(fake_only = True)
            fake_scores = discriminator(fake_batch) #Here we are not allowed to detach() since we need those gradients to train the generator/decoder.

            loss_G = Hinge_Loss_G(fake_scores)
            loss_G.backward()
            optimizer_dec.step()

            #logging
            epoch_loss_D += loss_D
            epoch_loss_G += loss_G

            total_real_score_D += real_scores.sum()
            total_fake_score_D += fake_scores_d.sum()

        
        

        avg_real_score_D = total_real_score_D/samples_per_epoch
        avg_fake_score_D = total_fake_score_D/samples_per_epoch
        avg_real_score_D_log.append(avg_real_score_D)
        avg_fake_score_D_log.append(avg_fake_score_D)

        loss_log_D.append(epoch_loss_D.item())
        loss_log_G.append(epoch_loss_G.item())



        

        logger.info(f"epoch loss is: D = {epoch_loss_D} | G = {epoch_loss_G}")
        logger.info(f"avg Discriminator predictions: real = {avg_real_score_D} | fake = {avg_fake_score_D}")

    ws.save_logs_GAN(experiment_directory, loss_log_D, loss_log_G, lr_log_D, lr_log_G, epoch)
    ws.save_latest(epoch, experiment_directory, decoder, "latest.pth",None, GAN = GAN_architecture)
    plot_logs(experiment_directory,show_lr = True, filename=os.path.join(experiment_directory, ws.logplot_filename), GAN = GAN_architecture)
            
            


