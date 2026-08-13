import torch
import logging
import socket
import tqdm
import time
import os
import sys

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

    #check if experiment has already been run before
    print(experiment_directory + "/ModelParameters")

    if os.path.isdir(experiment_directory + "/ModelParameters"):
        answer = input("The network has already been trained. Do you wish to retrain? (Y/N)")

        if answer.lower() == "y" or answer.lower() == "":
            print("continuing with training setup")
        if answer.lower() == "n":
            print("Stopping training...")
            sys.exit()
    
    #load experiment specs
    specs = ws.load_experiment_specifications(experiment_directory)
    disc_specs = specs["DiscriminatorSpecs"]
    logger.info(f"Reading experiment configuration from {experiment_directory}")
    logger.info("Experiment description: \n" + specs["Description"])
    GAN_architecture = specs["GANArchitecture"]

    #Determine decoder snapshot epochs
    snapshot_epochs = list(range(specs["SnapshotFrequency"],specs["NumEpochs"] + 1,specs["SnapshotFrequency"]))
    snapshot_epochs += specs["AdditionalSnapshots"]
    snapshot_epochs.sort()

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
    discriminator = ConvDiscriminator(disc_specs["n_nodes"], disc_specs["spectral_reg"]).to(device)

    #initialize optimizers
    optimizer_dec = torch.optim.Adam(decoder.parameters(),
                                      lr = specs["InitialLearningRates"]["decoder"])
    optimizer_disc = torch.optim.Adam(discriminator.parameters(),
                                      lr = specs["InitialLearningRates"]["discriminator"])
    
    #Batch size Variables and checking
    samples_per_batch = specs["SamplesPerBatch"]
    batch_per_epoch = specs["BatchPerEpoch"]
    samples_per_epoch_D = samples_per_batch * batch_per_epoch
    samples_per_epoch_G = specs["LearnRatio"] * samples_per_epoch_D

    if samples_per_epoch_D % 2 != 0:
            raise RuntimeError("samples_per_batch * batch_per_epoch must be divisible by 2 to ensure equal amounts of real and fake inputs for discriminator!")
    num_real_samples = batch_per_epoch*samples_per_batch // 2

    #Data Generation/Sampling
    real_sdf = real_sdf = CrossMsSDF(0.0)
    sampler = ConvGAN_SDF_Sampler(real_sdf, decoder, specs["DiscriminatorSpecs"]["n_nodes"], samples_per_batch, specs["SdfParameterBounds"], device)

    #Training stat logging
    loss_log_D = []
    loss_log_G = []
    lr_log_D = []
    lr_log_G = []
    #lat_mag_log = []
    timing_log = [] #UNUSED
    param_mag_log = {} #UNUSED

    avg_real_log = []
    avg_fake_log = []
    pred_accuracy_log = []

    decoder.train()
    discriminator.train()

    epoch = 1
    start_train = time.time()
    pbar = tqdm.trange(epoch, specs["NumEpochs"] + 1, desc="Training", smoothing=0)
    for epoch in pbar:
        start = time.time()

        #--------Logging--------
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        total_real_score_D = 0.0
        total_fake_score_D = 0.0

        correct_pred = 0.0

        #IMPLEMENT ADJUSTABLE LEARNING RATE!!
        lr_log_D.append(specs["InitialLearningRates"]["discriminator"])
        lr_log_G.append(specs["InitialLearningRates"]["decoder"])
   

        

        for batch in range(batch_per_epoch):
            real_batch, fake_batch = sampler.fetch_samples()

            #Train Discriminator
            real_scores = discriminator(real_batch)
            fake_scores_d = discriminator(fake_batch.detach()) #We need to detach the fake_batch so all our fake samples are treated as constant and our G gradients dont flow into our D gradient

            optimizer_disc.zero_grad()

            loss_D = Hinge_Loss_D(real_scores, fake_scores_d)
            loss_D.backward()

            optimizer_disc.step()
            
            epoch_loss_D += loss_D

            #Train Generator
            #Eventually turn off gradient calculation for discriminator here since they are not used -> eventual performance increase
            for i in range(specs["LearnRatio"]):
                fake_batch = sampler.fetch_samples(fake_only = True, decoder_clamp_val = specs["DecoderClampValue"])
                fake_scores = discriminator(fake_batch) #Here we are not allowed to detach() since we need those gradients to train the generator/decoder.

                optimizer_dec.zero_grad()

                loss_G = Hinge_Loss_G(fake_scores)
                loss_G.backward()

                optimizer_dec.step()

                epoch_loss_G += loss_G

            #--------Logging--------
            total_real_score_D += real_scores.sum().item()
            total_fake_score_D += fake_scores_d.sum().item()

            correct_pred += (real_scores > 0).float().sum().item() #Logit > 0 means real prediction. So this simply sums up all the correct Logit scores for real samples
            correct_pred += (fake_scores_d < 0).float().sum().item() #vice versa.

        if specs["LearnRatio"] > 1:
            epoch_loss_G /= specs["LearnRatio"]
        
        #--------Logging--------
        avg_real_pred = total_real_score_D/samples_per_epoch_D
        avg_fake_pred = total_fake_score_D/samples_per_epoch_D
        avg_real_log.append(avg_real_pred)
        avg_fake_log.append(avg_fake_pred)

        loss_log_D.append(epoch_loss_D.item())# type: ignore
        loss_log_G.append(epoch_loss_G.item())# type: ignore

        pred_accuracy = 100 * correct_pred / samples_per_epoch_D
        pred_accuracy_log.append(pred_accuracy)

        

        logger.info(f"Epoch loss is: D = {epoch_loss_D} | G = {epoch_loss_G}")
        logger.info(f"Avg. Discriminator predictions: real = {avg_real_pred} | fake = {avg_fake_pred} | Accuracy = {pred_accuracy}%" )
    
        if epoch in snapshot_epochs:
            save_snapshot(epoch, experiment_directory, decoder)


    ws.save_logs_GAN(experiment_directory, loss_log_D, loss_log_G, lr_log_D, lr_log_G, avg_real_log, avg_fake_log, pred_accuracy_log, epoch) # type: ignore
    ws.save_latest(epoch, experiment_directory, decoder, "latest.pth",None, GAN = GAN_architecture)
    plot_logs(experiment_directory,show_lr = True, filename=os.path.join(experiment_directory, ws.logplot_filename), GAN = GAN_architecture, snapshot_epochs = snapshot_epochs)
    plot_decoder_evolution(experiment_directory, snapshot_epochs)     
            
def save_snapshot(epoch, experiment_directory, decoder):
    ws.save_model(experiment_directory, "SnapshotE-" + str(epoch) + ".pth", decoder, epoch)