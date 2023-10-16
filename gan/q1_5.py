import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for discriminator.
    # loss_pt1 = E[D(fake_data)] - E[D(real_data)]
    # loss_pt2 = lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # loss = loss_pt1 + loss_pt2
    ##################################################################
    '''
    Wasserstein GAN with Gradient Penalty for Lipschitzness (WGAN-GP) loss for discriminator
    Improved Training of Wasserstein GANs (Gulrajani et al, 2017): https://arxiv.org/pdf/1704.00028.pdf
    Goal: Design a GAN objective function such that the generator G minimizes the Earth Mover (Wasserstein) distance between data x and generated distributions G(z).
    
    discrim_real: D(x) <-- Discriminator's predicted probability on real data
    discrim_fake: D(G(z)) <-- Discriminator's predicted probability on fake data
    discrim_interp: D(x_hat) <-- Discriminator's predicted probability on interpolated data
        x: real image
        x_hat: interpolated image
        z: random noise
        D: discriminator network
        G: generator network
    '''
    # Wassertein Critic Objective (Loss Function)
    loss_Wassertein = torch.mean(discrim_fake) - torch.mean(discrim_real) # E[D(G(z))] - E[D(x)] where E is 1/m * sum a.k.a. Expectation (Mean)

    # Gradient Penalty (Regularization) for Lipschitzness
    gradient_term = torch.autograd.grad(outputs=discrim_interp, # D(x_hat)
                                        inputs=interp, # x_hat
                                        grad_outputs=torch.ones_like(discrim_interp), # weights (scaling factor) that are multiplied to the gradients (d outputs/ d inputs) during backpropagation
                                        create_graph=True)[0] # gradient dD(x_hat)/x_hat where x_hat is interpolated_data
    l2_loss = torch.linalg.norm(gradient_term, dim=1, ord=2) # L2 norm
    gradient_penalty = lamb * (l2_loss - 1)**2 # gradient penalty loss
    loss_GP = torch.mean(gradient_penalty) # E[gradient_penalty] where E is 1/m * sum a.k.a. Expectation (Mean)

    loss = loss_Wassertein + loss_GP
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    ##################################################################
    '''
    Wasserstein GAN with Gradient Penalty (WGAN-GP) loss for generator
    Improved Training of Wasserstein GANs (Gulrajani et al, 2017): https://arxiv.org/pdf/1704.00028.pdf

    discrim_fake: D(G(z)) <-- Discriminator's predicted probability on fake data
        z: random noise
        G: generator network
        D: discriminator network
    '''
    loss = -torch.mean(discrim_fake) # -E[D(G(z))] where E is 1/m * sum a.k.a. Expectation (Mean)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
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
