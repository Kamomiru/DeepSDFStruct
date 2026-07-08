import torch
import torch.nn.functional as F

def Hinge_Loss_D(real_scores: torch.Tensor,
                 fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for the discriminator.

    Parameters
    ----------
    real_scores : (B,) or (B,1)
        Discriminator outputs for real samples.
    fake_scores : (B,) or (B,1)
        Discriminator outputs for generated samples.

    Returns
    -------
    torch.Tensor
        Scalar discriminator loss.
    """
    loss_real = F.relu(1.0 - real_scores).mean()
    loss_fake = F.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake

def Hinge_Loss_G(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for the generator.

    Parameters
    ----------
    fake_scores : (B,) or (B,1)
        Discriminator outputs for generated samples.

    Returns
    -------
    torch.Tensor
        Scalar generator loss.
    """
    return -fake_scores.mean()