import torch
import logging
import tqdm
import time
import os
 
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

def calc_lambda_relative(loss_GAN, loss_aux, alpha, eps=1e-8):
    if alpha == None: #if alpha is set to None, dont apply a weight to the losses
        return 1

    lambda_relative = abs((loss_GAN.detach()/(loss_aux.detach() + eps)) * (alpha/(1-alpha)))
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

    # NOTE: this warmup only ever builds a 1-D latent (`[[warmup_latent]]`).
    # Now that CodeLength can be > 1 (z + control code), that no longer
    # matches the decoder's expected input width. Fails fast with a clear
    # message rather than a confusing shape-mismatch error deep in the
    # decoder's first Linear layer. Extending this to sample a full
    # (z, c) latent -- e.g. random z + warmup_latent as c -- is still TODO.
    if specs["CodeLength"] != 1:
        raise NotImplementedError(
            "pretrain_decoder currently only supports CodeLength == 1 (a single "
            "scalar latent). It needs to be extended to build a full (z, c) latent "
            "vector before it can be used with the InfoGAN-style multi-dimensional "
            "latent setup."
        )

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
    plot_decoder_set(decoder, ax, code_dim=1, latent_size=specs["CodeLength"], device="cuda")
    plt.savefig(experiment_directory + f"/pretrainedDecoder{warmup_quality}.png")
 
    return decoder, loss_log


def save_checkpoint_GAN(
    tag,
    epoch,
    experiment_directory,
    decoder,
    discriminator,
    optimizer_dec,
    optimizer_disc,
    regressor=None,
    optimizer_reg=None,
):
    """
    Save a full, resumable GAN checkpoint under `tag` (e.g. "latest" or
    "SnapshotE-500"). Decoder, discriminator, (optional) regressor and their
    optimizers are each written as separate files, following the existing
    ModelParameters / OptimizerParameters split used elsewhere in workspace.py.
    """
    ws.save_model(experiment_directory, f"{tag}.pth", decoder, epoch)
    ws.save_model(experiment_directory, f"{tag}_disc.pth", discriminator, epoch)
    ws.save_optimizer(experiment_directory, f"{tag}_optimizer_dec.pth", optimizer_dec, epoch)
    ws.save_optimizer(experiment_directory, f"{tag}_optimizer_disc.pth", optimizer_disc, epoch)
 
    if regressor is not None:
        ws.save_model(experiment_directory, f"{tag}_reg.pth", regressor, epoch)
    if optimizer_reg is not None:
        ws.save_optimizer(experiment_directory, f"{tag}_optimizer_reg.pth", optimizer_reg, epoch)
 
 
def load_checkpoint_GAN(
    tag,
    experiment_directory,
    decoder,
    discriminator,
    optimizer_dec,
    optimizer_disc,
    device,
    regressor=None,
    optimizer_reg=None,
):
    """
    Load a full GAN checkpoint saved under `tag`, in-place, into the given
    decoder/discriminator/(regressor)/optimizers. Returns the epoch the
    checkpoint was saved at (i.e. the last *completed* epoch of that run).
    """
    epoch = ws.load_model_parameters(experiment_directory, tag, decoder, device)
    ws.load_model_parameters(experiment_directory, f"{tag}_disc", discriminator, device)
    ws.load_optimizer(experiment_directory, f"{tag}_optimizer_dec", optimizer_dec, device)
    ws.load_optimizer(experiment_directory, f"{tag}_optimizer_disc", optimizer_disc, device)
 
    if regressor is not None:
        ws.load_model_parameters(experiment_directory, f"{tag}_reg", regressor, device)
    if optimizer_reg is not None:
        ws.load_optimizer(experiment_directory, f"{tag}_optimizer_reg", optimizer_reg, device)
 
    return epoch
 
 
def load_previous_logs_GAN(experiment_directory):
    """
    Load previously saved GAN logs (Logs.pth) into a dict of plain lists,
    so a continued run can extend them and later plots show the full history.

    Regressor-related logs default to empty lists if the previous run
    didn't use a regressor.
    """

    logs = torch.load(
        os.path.join(experiment_directory, "logs.pth"),
        weights_only=False
    )

    return {
        "loss_log_D": logs.get("loss_D", []),
        "loss_log_G": logs.get("loss_G", []),
        "lr_log_D": logs.get("lr_log_D", []),
        "lr_log_G": logs.get("lr_log_G", []),
        "disc_avg_real_log": logs.get("avg_real_pred", []),
        "disc_avg_fake_log": logs.get("avg_fake_pred", []),
        "disc_pred_accuracy_log": logs.get("pred_accuracy", []),

        # Regressor-related logs
        "RMSE_error_log_R": logs.get("RMSE_error_log_R", []),
        "lr_log_R": logs.get("lr_log_R", []),
        "loss_log_G_GAN": logs.get("loss_G_GAN", []),
        "loss_log_G_reg": logs.get("loss_G_reg", []),
        "loss_log_G_reg_base": logs.get("loss_G_reg_base", []),

        # Useful if you need the previous epoch
        "epoch": logs.get("epoch", 0),
    }
def get_lr_single(schedule, epoch):
    schedule_type = schedule["Type"]
    lr = 0.0
    if schedule_type == "Constant":
        lr = schedule["InitialLr"]
        return lr
    elif schedule_type == "FactorStep":
        lr = schedule["InitialLr"] * (schedule["Factor"] ** (epoch // schedule["Interval"]))
        return lr
    elif schedule_type == "ConstantStep":
        raise NotImplementedError(f"ConstantStep Lr schedule not implemented yet!")
    elif schedule_type == "Custom":
        entries = schedule["Schedule"]

        if not entries:
            raise ValueError("Custom LR schedule cannot be empty.")

        # Find the most recent entry whose epoch <= current epoch
        lr = entries[0]["lr"] #first initial Lr

        for entry in entries:
            if epoch >= entry["epoch"]:
                lr = entry["lr"]
            else: #stop updating lr if epoch is not >= current epoch
                break

        return lr

    else:
        raise ValueError(f"Unknown LR schedule type: {schedule_type}")


def get_lr_all(specs, epoch):
    lr_schedules = specs["LearningRateSchedule"]

    lr_G = get_lr_single(lr_schedules["Generator"], epoch)
    lr_D = get_lr_single(lr_schedules["Discriminator"], epoch)

    lr_R = None
    if specs["UseRegressor"]:
        lr_R = get_lr_single(lr_schedules["Regressor"], epoch)

    return lr_G, lr_D, lr_R

def update_lr(opt_G, opt_D, opt_R, specs, epoch):

    lr_G, lr_D, lr_R = get_lr_all(specs, epoch)

    opt_G.param_groups[0]["lr"] = lr_G
    opt_D.param_groups[0]["lr"] = lr_D
    if opt_R is not None:
        opt_R.param_groups[0]["lr"] = lr_R

    return lr_G, lr_D, lr_R