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
from DeepSDFStruct.deep_sdf.networks.deep_sdf_classifier import ConvClassifier
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
    if os.path.isdir(experiment_directory + "/ModelParameters"):
        while True:
            answer = input(
                "The network has already been trained. "
                "Do you wish to retrain? (Y/N): "
            ).strip().lower()

            if answer in ("y", ""):
                print("Continuing with training setup...")
                break

            if answer == "n":
                print("Stopping training...")
                sys.exit(0)

            print("Please enter Y or N.")
    
    #load experiment specs
    specs = ws.load_experiment_specifications(experiment_directory)
    disc_specs = specs["DiscriminatorSpecs"]
    logger.info(f"Reading experiment configuration from {experiment_directory}")
    logger.info("Experiment description: \n" + specs["Description"])
    GAN_architecture = specs["GANArchitecture"]
    UseClassifier = specs["UseClassifier"]

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

    if samples_per_epoch_D % 2 != 0:
            raise RuntimeError("samples_per_batch * batch_per_epoch must be divisible by 2 to ensure equal amounts of real and fake inputs for discriminator!")

    #Data Generation/Sampling
    real_sdf = CrossMsSDF(0.0)
    sampler = ConvGAN_SDF_Sampler(real_sdf, decoder, specs["DiscriminatorSpecs"]["n_nodes"], samples_per_batch, specs["SdfParameterBounds"], device, add_latent = UseClassifier)

    #Training stat logging
    loss_log_D = []
    loss_log_G = []
    lr_log_D = []
    lr_log_G = []


    disc_avg_real_log = []
    disc_avg_fake_log = []
    disc_pred_accuracy_log = []

    decoder.train()
    discriminator.train()

    #OPTIONAL: initialize parameter classifier
    #type annotation so pylance does'nt constantly throw errors. Would work without this!
    classifier: ConvClassifier | None = None
    optimizer_cla: torch.optim.Optimizer | None = None
    loss_log_C: list = []
    loss_log_G_GAN: list = []
    loss_log_G_cla: list = []
    lr_log_C: list = []  
    RMSE_error_C: torch.Tensor = torch.tensor(0.0)
    RMSE_error_log_C: list = []
    epoch_loss_C: float = 0.0
    epoch_loss_G_GAN: float = 0.0
    epoch_loss_G_cla: float = 0.0
    alpha_loss = specs["ClassifierLossRatio"]
    lambda_relative: float = 1.0

    #initialization
    if specs["UseClassifier"]:
        classifier = ConvClassifier(disc_specs["n_nodes"], disc_specs["spectral_reg"], specs["CodeLength"]).to(device)
        optimizer_cla = torch.optim.Adam(classifier.parameters(),
                                         lr = specs["InitialLearningRates"]["classifier"])
        classifier.train()

    epoch = 1
    start_train = time.time()
    pbar = tqdm.trange(epoch, specs["NumEpochs"] + 1, desc="Training", smoothing=0)
    for epoch in pbar:
        start = time.time()

        #--------Logging--------
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_loss_G_GAN = 0.0
        epoch_loss_G_cla = 0.0

        total_real_score_D = 0.0
        total_fake_score_D = 0.0

        correct_disc_pred = 0.0

        #IMPLEMENT ADJUSTABLE LEARNING RATE!!
        lr_log_D.append(specs["InitialLearningRates"]["discriminator"])
        lr_log_G.append(specs["InitialLearningRates"]["decoder"])

        if classifier is not None:
            epoch_loss_C = 0.0
            lr_log_C.append(specs["InitialLearningRates"]["classifier"])
            
            

        

        for batch in range(batch_per_epoch):
            if classifier is not None:
                real_batch, fake_batch, latent_parameters = sampler.fetch_samples() #type: ignore
            else: 
                real_batch, fake_batch = sampler.fetch_samples() #type: ignore

            #Train Discriminator
            real_scores = discriminator(real_batch)
            fake_scores_d = discriminator(fake_batch.detach()) #We need to detach the fake_batch so all our fake samples are treated as constant and our G gradients dont flow into our D gradient

            optimizer_disc.zero_grad()

            loss_D = Hinge_Loss_D(real_scores, fake_scores_d)
            loss_D.backward()

            optimizer_disc.step()
            
            epoch_loss_D += loss_D.item()

            #Train Classifier
            if classifier is not None and optimizer_cla is not None:
                freeze_network(classifier, False)
                pred_latent_real = classifier(real_batch)

                optimizer_cla.zero_grad()

                loss_C = torch.nn.functional.smooth_l1_loss(pred_latent_real, latent_parameters)
                RMSE_error_C = torch.sqrt(torch.mean((pred_latent_real - latent_parameters) ** 2)) #Root Mean Square Error for classifier training

                loss_C.backward()

                optimizer_cla.step()

                epoch_loss_C += loss_C.item()

            #Train Generator
            #Eventually turn off gradient calculation for discriminator here since they are not used -> eventual performance increase
            for i in range(specs["LearnRatio"]):

                if classifier is not None:
                    fake_batch, latent_parameters_fake = sampler.fetch_samples(fake_only = True, decoder_clamp_val = specs["DecoderClampValue"])
                else:
                    fake_batch = sampler.fetch_samples(fake_only = True, decoder_clamp_val = specs["DecoderClampValue"])
                
                fake_scores = discriminator(fake_batch) #Here we are not allowed to detach() since we need those gradients to train the generator/decoder.

                optimizer_dec.zero_grad()

                loss_G_GAN = Hinge_Loss_G(fake_scores)

                #additional classifier loss
                if classifier is not None:
                    freeze_network(classifier) #type: ignore
                    pred_latent_fake = classifier(fake_batch)


                    loss_G_cla = torch.nn.functional.smooth_l1_loss(pred_latent_fake, latent_parameters_fake)

                    lambda_relative = calc_lambda_relative(loss_G_GAN, loss_G_cla, alpha_loss)

                    loss_G_cla *= lambda_relative

                    loss_G = loss_G_GAN + loss_G_cla


                    epoch_loss_G_cla += loss_G_cla.item()
                else:
                    loss_G = loss_G_GAN

                epoch_loss_G_GAN += loss_G_GAN.item()

                loss_G.backward()

                optimizer_dec.step()

                epoch_loss_G += loss_G.item() #Implement logging of total and partial loss for generator!

            #--------Logging--------
            total_real_score_D += real_scores.sum().item()
            total_fake_score_D += fake_scores_d.sum().item()

            correct_disc_pred += (real_scores > 0).float().sum().item() #Logit > 0 means real prediction. So this simply sums up all the correct Logit scores for real samples
            correct_disc_pred += (fake_scores_d < 0).float().sum().item() #vice versa.

        if specs["LearnRatio"] > 1:
            epoch_loss_G /= specs["LearnRatio"]
            epoch_loss_G_GAN /= specs["LearnRatio"]
            epoch_loss_G_cla /= specs["LearnRatio"]
        
        #--------Logging--------
        avg_real_pred = total_real_score_D/samples_per_epoch_D
        avg_fake_pred = total_fake_score_D/samples_per_epoch_D
        disc_avg_real_log.append(avg_real_pred)
        disc_avg_fake_log.append(avg_fake_pred)

        loss_log_D.append(epoch_loss_D)
        loss_log_G.append(epoch_loss_G)

        pred_accuracy = 100 * correct_disc_pred / samples_per_epoch_D
        disc_pred_accuracy_log.append(pred_accuracy)

        #classifier logging
        if classifier is not None:
            loss_log_C.append(epoch_loss_C)
            RMSE_error_log_C.append(RMSE_error_C.item())
            loss_log_G_cla.append(epoch_loss_G_cla)
            loss_log_G_GAN.append(epoch_loss_G_GAN)
            


        

        logger.info(f"Epoch loss is: D = {epoch_loss_D} | G = {epoch_loss_G} | C = {epoch_loss_C}")
        logger.info(f"Partial Generator loss is: G_GAN = {epoch_loss_G_GAN} | G_cla = {epoch_loss_G_cla} (λrel={lambda_relative})")
        logger.info(f"Avg. Discriminator predictions: real = {avg_real_pred} | fake = {avg_fake_pred} | Accuracy = {pred_accuracy}%" )
        logger.info(f"Classifier Metrics: RMSE = {RMSE_error_C}")
    
        if epoch in snapshot_epochs:
            save_snapshot(epoch, experiment_directory, decoder)

    #Store all Logs and create plots
    #logger class waere hier schon mal gut gewesen :/
    ws.save_logs_GAN(experiment_directory, loss_log_D, loss_log_G, lr_log_D, lr_log_G, disc_avg_real_log, disc_avg_fake_log, disc_pred_accuracy_log, loss_log_C, RMSE_error_log_C, lr_log_C, loss_log_G_GAN, loss_log_G_cla, epoch) # type: ignore
    ws.save_latest(epoch, experiment_directory, decoder, "latest.pth",None, GAN = GAN_architecture)
    plot_logs(experiment_directory,show_lr = True, filename=os.path.join(experiment_directory, ws.logplot_filename), GAN = GAN_architecture, snapshot_epochs = snapshot_epochs)
    plot_decoder_evolution(experiment_directory, snapshot_epochs)
    plot_decoder_latent_effect(experiment_directory)  
            
def save_snapshot(epoch, experiment_directory, decoder):
    ws.save_model(experiment_directory, "SnapshotE-" + str(epoch) + ".pth", decoder, epoch)

def freeze_network(network, bool = True):
    for param in network.parameters():
        param.requires_grad = not bool

def calc_lambda_relative(loss_GAN, loss_cla, alpha, eps=1e-8):
    if alpha == None: #if alpha is set to None, dont apply a weight to the losses
        return 1

    lambda_relative = (loss_GAN.detach()/(loss_cla.detach() + eps)) * (alpha/(1-alpha))
    #maybe also clamp lambda_relative?
    return lambda_relative