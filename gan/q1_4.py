import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    '''
    LSGAN (Least Squares GAN) loss for discriminator (https://arxiv.org/pdf/1611.04076.pdf)
    discrim_real: D(x)
    discrim_fake: D(G(z))
        x: real image
        z: random noise
        D: discriminator network
        G: generator network
    '''
    def MSE_loss(input, target):
        return torch.mean((input - target)**2)

    real_label = torch.ones_like(discrim_real) # [1, 1, 1, ...]; coined as "b" in the paper
    fake_label = torch.zeros_like(discrim_fake) # [0, 0, 0, ...]; coined as "a" in the paper
    loss = 0.5 * MSE_loss(input=discrim_real, target=real_label) + 0.5 * MSE_loss(input=discrim_fake, target=fake_label)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    '''
        discrim_fake: D(G(z))
        z: random noise
        G: generator network
        D: discriminator network
    '''
    def MSE_loss(input, target):
        return torch.mean((input - target)**2)

    fake_real_label = torch.ones(discrim_fake) # [1, 1, 1, ...]; coined as "c" in the paper (labels that Generator wants Discriminator to believe for fake data)
    loss = 0.5* MSE_loss(input=discrim_fake, target=fake_real_label)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
