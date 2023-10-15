import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    # Generate 100 samples of 128-dim vectors.
    samples = torch.zeros(100, 128)

    # Linearly interpolate the first two dimensions of the sample vector between -1 and 1 for 10 steps.
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    x_grid, y_grid = x_grid.reshape(-1), y_grid.reshape(-1)
    samples[:, 0] = x_grid
    samples[:, 1] = y_grid

    # Forward the samples through the generator.
    samples = gen.forward_given_samples(samples)

    # Save out an image holding all 100 samples.
    torchvision.utils.save_image(samples, path)
    
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
