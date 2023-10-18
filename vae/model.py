import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    """
    Convolution Output Size Calculation (Used here for downsampling feature map size):
        H_out = (H_in - K + 2P)/S + 1
        W_out = (W_in - K + 2P)/S + 1

    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))       size: (B,C=32.H,W) --> (B,C=32,H,W)
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))      size: (B,C=64,H,W) --> (B,C=64,H/2,W/2)
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))     size: (B,C=64,H/2,W/2) --> (B,C=128,H/4,W/4)
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))    size: (B,C=128,H/4,W/4) --> (B,C=256,H/8,W/8)
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        self.convs = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                *[nn.Sequential(nn.ReLU(), nn.Conv2d(in_channels=32*2**i, out_channels=64*2**i, kernel_size=3, stride=2, padding=1)) for i in range(3)]
                                )
        H, W = input_shape[-2]//8, input_shape[-1]//8 # convs output image shape
        convs_out_dim = 256 * (H) * (W) # flattened convs output dimension (C*H_reduced*W_reduced)
        self.fc = nn.Linear(convs_out_dim, self.latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        x = self.convs(x) # (B,C,H_reduced,W_reduced)
        x = x.view(x.shape[0], -1) # (B,C*H_reduced*W_reduced)
        x = self.fc(x) # (B,latent_dim)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        H, W = input_shape[-2]//8, input_shape[-1]//8 # convs output image shape
        convs_out_dim = 256 * (H) * (W) # flattened convs output dimension (C*H_reduced*W_reduced)
        self.fc1 = nn.Linear(in_features=convs_out_dim, out_features=self.latent_dim)
        self.fc2 = nn.Linear(in_features=convs_out_dim, out_features=self.latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        x = self.convs(x) # (B,C,H_reduced,W_reduced)
        x = x.view(x.shape[0], -1) # (B,C*H_reduced*W_reduce)
        mu = self.fc1(x) # mean; (B,latent_dim)
        log_std = self.fc2(x) # log(std_dev); (B,latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Transposed Convolution Output Size Calculation (Used here for upsampling feature map size):
        H_out = (H_in - 1)*S + K - 2P
        W_out = (W_in - 1)*S + K - 2P

    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))   Size: (B,C=256,H,W) --> (B,C=128,H*2,W*2)
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))    Size: (B,C=128,H*2,W*2) --> (B,C=64,H*4,W*4)
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))     Size: (B,C=64,H*4,W*4) --> (B,C=32,H*8,W*8)
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))               Size: (B,C=32,H*8,W*8) --> (B,C=3,H*8,W*8)
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        self.base_size = 256 * (output_shape[-2]//8) * (output_shape[-1]//8) # flattened convs output dimension (C*H_reduced*W_reduced)
        self.fc = nn.Linear(self.latent_dim, self.base_size)
        self.deconvs = nn.Sequential(
                                    *[nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(in_channels=256//2**i, out_channels=128//2**i, kernel_size=4, stride=2, padding=1)) for i in range(3)],
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
                                    )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        x = self.fc(z) # (B,base_size)
        x = x.reshape(x.shape[0], 256, self.output_shape[-2]//8, self.output_shape[-1]//8) # (B,C,H_reduced,W_reduced)
        x = self.deconvs(x) # (B,C,H,W)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
