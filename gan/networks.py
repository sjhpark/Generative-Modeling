import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
        ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding)
        self.upscale_factor = upscale_factor

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        # (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        # Input x: shape (B,C,H,W)
        
        # 1. Repeat x channel-wise upscale factor^2 times
        # shape (B,C,H,W) -> (B,C*r^2,H,W) where r is an upscale factor
        x = x.repeat(1, self.upscale_factor**2, 1, 1) # (B,C*r^2,H,W)

        # 2. Use torch.nn.PixelShuffle to form an output of dimension.
        # Rearrange elements in a tensor of shape (B,C*r^2,H,W) -> (B,C,H*r,W*r) where r is an upscale factor.
        # This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
        pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        x = pixel_shuffle(x) # (B,C,H*r,W*r)

        # 3. Apply convolution and return output.
        # (B,C,H*r,W*r) -> CONV LAYER -> (B, C, [(H*r−K+2P)/S]+1, [(W*r−K+2P)/S]+1) where K is kernel size, P is padding, S is stride
        x = self.conv(x) # (B, n_filters, [(H*r−K+2P)/S]+1, [(W*r−K+2P)/S]+1)

        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
        ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding)
        self.downscale_ratio = downscale_ratio

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # (batch, channel, downscale_factor^2, height, width)
        # 2. Then split channel-wise into
        # (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        # and return the output
        ##################################################################
        # Input x: shape (B,C,H,W)

        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # Rearrange elements in a tensor of shape (B,C,H,W) -> (B,C*r^2,H/r,W/r)
        # Then rearrange elements in a tensor of shape (B,C*r^2,H/r,W/r) -> (B,C,r^2,H/r,W/r)
        # downscale_factor = self.upscale_factor
        downscale_factor = self.downscale_ratio
        pixel_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)
        x = pixel_unshuffle(x) # (B,C*r^2,H/r,W/r)
        x = x.reshape(x.shape[0], x.shape[1]//downscale_factor**2, downscale_factor**2, x.shape[2], x.shape[3]) # (B,C,r^2,H/r,W/r)

        # 2. Rearrange shape from (B,C,r^2,H/r,W/r) -> (r^2,B,C,H/r,W/r).
        x = x.permute(2,0,1,3,4) # (r^2,B,C,H/r,W/r)

        # 3. Take the average across dimension 0, and apply convolution and return the output.
        x = torch.mean(x, dim=0) # (B,C,H/r,W/r)
        x = self.conv(x) # (B, n_filters, [(H*r−K+2P)/S]+1, [(W*r−K+2P)/S]+1)

        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
                                    nn.BatchNorm2d(num_features=input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(num_features=n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    UpSampleConv2D(input_channels=n_filters, n_filters=n_filters, kernel_size=3, padding=1)
                                    )
        self.upsample_residual = UpSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to upsample the residual before adding it
        # to the layer output.
        ##################################################################
        x = self.layers(x) + self.upsample_residual(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    DownSampleConv2D(input_channels=n_filters, n_filters=n_filters, kernel_size=3, padding=1)
                                    )
        self.downsample_residual = DownSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to downsample the residual before adding
        # it to the layer output.
        ##################################################################
        x = self.layers(x) + self.downsample_residual(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1)
                                    )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the conv layers. Don't forget the residual
        # connection!
        ##################################################################
        x = self.layers(x) + x
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.starting_image_size = starting_image_size

        self.dense = nn.Linear(in_features=128, out_features=2048, bias=True)

        self.layers = nn.Sequential(
                                    *[ResBlockUp(input_channels=128, n_filters=128, kernel_size=3) for _ in range(3)],
                                    nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh()
                                    )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # TODO 1.1: Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        x = self.dense(z) # (n_samples, 2048)
        x = x.view(-1, 128, self.starting_image_size, self.starting_image_size) # (B=n_samples, C=128, H=img_size, W=img_size)
        x = self.layers(x) # (B=n_samples, C=3, H=img_size*2^3, W=img_size*2^3)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, n_samples:int=1024):
        ##################################################################
        # TODO 1.1: Generate n_samples latents and forward through the
        # network.
        ##################################################################
        z = torch.randn(n_samples, 128).cuda() # (B=n_samples, 128); n_samples of noise from a Gaussian/Normal distribution (mu=0, sigma=1)
        return self.forward_given_samples(z) # (B=n_samples, C=3, H=img_size, W=img_size)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.layers = nn.Sequential(
                                    ResBlockDown(input_channels=3, n_filters=128, kernel_size=3),
                                    ResBlockDown(input_channels=128, n_filters=128, kernel_size=3),
                                    ResBlock(input_channels=128, n_filters=128, kernel_size=3),
                                    nn.ReLU()
                                    )
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################
        x = self.layers(x) # (B=n_samples, C=128, H=[(H−K+2P)/S]+1=[(4-3+2*1)/1]+1=4, W=[(W−K+2P)/S]+1=[(4-3+2*1)/1]+1=4)
        x = torch.sum(x, dim=(2,3)) # (n_samples, C=128)
        x = self.dense(x) # (B=n_samples, 1)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
