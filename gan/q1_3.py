import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(discrim_real, discrim_fake, discrim_interp, interp, lamb):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    '''
    discrim_real: D(x) <-- Discriminator's predicted probability on real data
    discrim_fake: D(G(z)) <-- Discriminator's predicted probability on fake data
        x: real image
        z: random noise
        D: discriminator network
        G: generator network
    '''
    def sigmoid(x):
        '''
        sigmoid with clamping to avoid either vanishing or exploding gradients
        '''
        eps = 1e-8 # epsilon to avoid NaN
        return torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)

    def BCE_loss_with_logits(input, target):
        loss = -torch.mean(
                        target * torch.log(sigmoid(input)) +
                        (1 - target) * torch.log(1 - sigmoid(input))
                        )
        return loss
    
    real_label = torch.ones_like(discrim_real) # [1, 1, 1, ...]
    fake_label = torch.zeros_like(discrim_fake) # [0, 0, 0, ...]
    loss_real = BCE_loss_with_logits(input=discrim_real, target=real_label)
    loss_fake = BCE_loss_with_logits(input=discrim_fake, target=fake_label)

    loss = loss_real + loss_fake
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    '''
    discrim_fake: D(G(z)) <-- Discriminator's predicted probability on fake data
        z: random noise
        G: generator network
        D: discriminator network
    '''
    def sigmoid(x):
        '''
        sigmoid with clamping to avoid either vanishing or exploding gradients
        '''
        eps = 1e-8 # epsilon to avoid NaN
        return torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)

    def BCE_loss_with_logits(input, target):
        loss = -torch.mean(
                        target * torch.log(sigmoid(input)) +
                        (1 - target) * torch.log(1 - sigmoid(input))
                        )
        return loss
    
    real_label = torch.ones_like(discrim_fake) # [1, 1, 1, ...]
    loss = BCE_loss_with_logits(input=discrim_fake, target=real_label)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
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
