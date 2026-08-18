import torch
import logging
import tqdm
import time
 
import DeepSDFStruct
import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.sdf_primitives import CrossMsSDF
from DeepSDFStruct.deep_sdf.plotting import plot_decoder_set

import matplotlib.pyplot as plt
 
logger = logging.getLogger(DeepSDFStruct.__name__)

def save_snapshot(epoch, experiment_directory, decoder):
    ws.save_model(experiment_directory, "SnapshotE-" + str(epoch) + ".pth", decoder, epoch)

def freeze_network(network, bool = True):
    for param in network.parameters():
        param.requires_grad = not bool

def calc_lambda_relative(loss_GAN, loss_cla, alpha, eps=1e-8):
    if alpha == None: #if alpha is set to None, dont apply a weight to the losses
        return 1

    lambda_relative = abs((loss_GAN.detach()/(loss_cla.detach() + eps)) * (alpha/(1-alpha)))
    #maybe also clamp lambda_relative?
    return lambda_relative

def pretrain_decoder(experiment_directory, warmup_latent=0.5, device=None):
    """
    Warm up / pretrain the decoder on a single fixed shape before adversarial
    training. Fits the decoder (conditioned on `warmup_latent`) to CrossMsSDF
    at that same parameter value, using freshly sampled random points in
    [-1, 1]^3 every iteration and a clamped L1 loss (DeepSDF-paper style).
 
    Expects a "WarmupSpecs" block in the experiment's specs.json, e.g.:
 
        "WarmupSpecs": {
            "NumIterations": 5000,
            "SamplesPerIteration": 4096,
            "ClampDelta": 0.1,
            "LearningRate": 1e-4,
            "LogFrequency": 100
        }
    """

    #loading experiment specs
    experiment_directory = str(experiment_directory)
 
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    specs = ws.load_experiment_specifications(experiment_directory)
    warmup_specs = specs["WarmupSpecs"]
    samples_per_iteration = warmup_specs["SamplesPerIteration"]
    lr = warmup_specs["LearningRate"]
    log_frequency = warmup_specs.get("LogFrequency", 100)
    clamp_delta = None
    if specs["DecoderClampValue"] is not None:
        clamp_delta = specs["DecoderClampValue"]

    num_iterations = 0
    warmup_quality = warmup_specs["WarmupQuality"]
    if warmup_quality == "Minimal":
        num_iterations = 100
    elif warmup_quality == "Light":
        num_iterations = 500
    elif warmup_quality == "Standard":
        num_iterations = 1000
    elif warmup_quality == "Extreme":
        num_iterations = 5000
    else:
        raise ValueError(f"WarmupQuality has to be set to one of: [ Minimal | Light | Standard | Extreme ]\nGot: {warmup_quality}")

 
    logger.info(
        f"Warming up decoder from {experiment_directory} on CrossMsSDF(latent={warmup_latent})"
    )
 
    # decoder
    decoder = ws.init_decoder(specs, device, data_parallel=False).to(device)
    decoder.train()
 
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
 
    # fixed target shape for warmup
    real_sdf = CrossMsSDF(0.0)
    real_sdf.setRadius(warmup_latent)
 
    # latent code the decoder is conditioned on (assumes CodeLength == 1)
    # TODO: if training on SDFs with multidimensional latent vectors is needed, this needs to be changed
    latent = torch.tensor([[warmup_latent]], device=device)
 
    loss_log = []
    start = time.time()
 
    pbar = tqdm.trange(1, num_iterations + 1, desc="Warmup", smoothing=0)
    for iteration in pbar:
        xyz = (torch.rand(samples_per_iteration, 3, device=device) * 2.0) - 1.0
 
        with torch.no_grad():
            target_sdf = real_sdf.forward(xyz).view(-1, 1)
 
        pred_sdf = decoder.forward_with_latent(latent, xyz)

        if clamp_delta is not None:
            pred_sdf = torch.clamp(pred_sdf, -clamp_delta, clamp_delta)
 
        loss = torch.nn.functional.l1_loss(pred_sdf, target_sdf)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        loss_log.append(loss.item())
        pbar.set_postfix(loss=loss.item())
 
        if iteration % log_frequency == 0:
            logger.info(
                f"[Warmup] iteration {iteration}/{num_iterations} | loss = {loss.item():.6f}"
            )
 
    logger.info(
        f"Warmup finished in {time.time() - start:.1f}s | final loss = {loss_log[-1]:.6f}"
    )
 
    ws.save_model(experiment_directory, "warmup.pth", decoder, num_iterations)
    fig, ax = plt.subplots(1,3)
    plot_decoder_set(decoder, ax, device="cuda")
    plt.savefig(experiment_directory + f"/pretrainedDecoder{warmup_quality}.png")
 
    return decoder, loss_log