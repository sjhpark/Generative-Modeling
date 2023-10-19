import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (cosine_beta_schedule,default,
                    extract,unnormalize_to_zero_to_one)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        # beta_t
        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        # alpha_t
        alphas = 1. - self.betas # e.g. [1,2,3,4,5]
        ##################################################################
        # TODO 3.1: Compute the cumulative products for current and
        # previous timesteps.
        ##################################################################
        # alpha_bar_t: Cumulative product of alphas from 0 to t
        self.alphas_cumprod = alphas.cumprod(dim=0) # e.g. [1, 2, 6, 24, 120]
        self.device = self.alphas_cumprod.device

        # alpha_bar_{t-1}: Cumulative product of alphas from 0 to t-1
        self.alphas_cumprod_prev = torch.concat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0) # e.g. [1, 1, 2, 6, 24]
        
        ##################################################################
        # TODO 3.1: Pre-compute values needed for forward process.
        ##################################################################
        '''
        x_0_hat = 1/sqrt(alpha_bar_t) * x_t - [1/sqrt(alpha_bar_t) * sqrt(1 - alpha_bar_t)] * eps_t
            x_0: starting image
            x_0_hat: predicted starting image
            x_t: current noised image at timestamp t
            alpha_bar_t: cumulative product of alpha_t from 0 to t
            eps_t <-- f(x_t, t): predicted noise at timestamp t
                f(): denoising network (takes noised image & timestamp as inputs) to predict noise at timestamp t
        '''
        # Coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1 / torch.sqrt(self.alphas_cumprod)

        # Coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -torch.sqrt(1 - self.alphas_cumprod) / torch.sqrt(self.alphas_cumprod)
        
        ##################################################################
        # TODO 3.1: Compute the coefficients for the mean.
        ##################################################################
        '''
        mu_tilde = [sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)] * x_t + [sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)] * x_hat_0
            mu_tilde = mean of Gaussian Distribution for sampling x_{t-1} given x_t and x_hat_0
            alpha_bar_t: cumulative product of alpha_t from 0 to t
            alpha_bar_{t-1}: cumulative product of alpha_t from -1 to t-1
            beta_t: diffusion rate at timestamp t
            x_hat_0: predicted starting image
        '''
        # Coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)

        # Coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        ##################################################################
        # TODO 3.1: Compute posterior variance.
        ##################################################################
        '''
        sigma^2_t = [(1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)] * beta_t
            sigma^2_t: variance of Gaussian Distribution for sampling x_{t-1} given x_t and x_hat_0
            alpha_bar_t: cumulative product of alpha_t from 0 to t
            alpha_bar_{t-1}: cumulative product of alpha_t from -1 to t-1
            beta_t: diffusion rate at timestamp t
        '''
        # Variance of posterior conditional distribution q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = (1 - self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0.
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 \
                            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        ##################################################################
        # TODO 3.1: Given a noised image x_t, predict x_0 and the additive
        # noise to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py. See
        # get_posterior_parameters() for usage examples.
        ##################################################################
        '''
        eps_t = f(x_t, t)
            eps_t: predicted noise at timestamp t
            f(): denoising network (takes noised image & timestamp as inputs) to predict noise at timestamp t
                x_t: current noised image at timestamp t
                t: timestamp
        x_hat_0 = 1/sqrt(alpha_bar_t) * x_t - [1/sqrt(alpha_bar_t) * sqrt(1 - alpha_bar_t)] * eps_t
            x_0: starting image
            x_0_hat: predicted starting image
            x_t: current noised image at timestamp t
            alpha_bar_t: cumulative product of alpha_t from 0 to t
        '''
        # Predicted noise at timestamp t
        eps_t = self.model(x=x_t, time=t) # eps_t <-- f(x_t, t) where f is the denoising network

        # Predicted starting image
        x_hat_0 = extract(self.x_0_pred_coef_1, t, x_t.shape) * x_t \
                    + extract(self.x_0_pred_coef_2, t, x_t.shape) * eps_t
        x_hat_0 = x_hat_0.clamp(min=-1, max=1)
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return (eps_t, x_hat_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        ##################################################################
        # TODO 3.1: Given x at timestep t, predict the denoised image at
        # x_{t-1}, and return the predicted starting image.
        # noise to predict the additive noise, use the denoising model.
        # Hint: To do this, you will need a predicted x_0.
        ##################################################################
        '''
        x_hat_0 = 1/sqrt(alpha_bar_t) * x_t - [1/sqrt(alpha_bar_t) * sqrt(1 - alpha_bar_t)] * eps_t
            x_0: starting image
            x_0_hat: predicted starting image
            x_t: current noised image at timestamp t
            alpha_bar_t: cumulative product of alpha_t from 0 to t
        x_{t-1} = mu_tilde_t + sigma_t * z
            x_{t-1}: denoised image at timestamp t-1 given x_t and x_0 (b/c denoising process is backward from T to 0)
            mu_tilde_t: mean of Posterior Conditional Distribution q(x_{t-1} | x_t, x_0)
            sigma_t: standard deviation of Posterior Conditional Distribution q(x_{t-1} | x_t, x_0)
            z: noise
        '''
        # Predicted noise & Predicted starting image
        eps_t, x_hat_0 = self.model_predictions(x, t)
        
        # Mean, Variance, and Clipped Log Variance of Posterior Conditional Distribution q(x_{t-1} | x_t, x_0) at timestamp t
        mu_tilde_t, var_t, logvar_clipped_t = self.get_posterior_parameters(x_hat_0, x, t)

        # Reparameterization Trick
        def reparameterize(mu, var, t):
            '''
            Reparameterization trick to sample from N(mu, std)
                N(mu, std): Normal Distribution
            '''
            std = torch.sqrt(var) # std = sigma = standard deviation of Gaussian Distribution

            # Sample a noise vector if t > 0 otherwise, = 0
            if t.min() > 0:
                z = torch.randn_like(mu) # noise sampled from N(0,1) where N(0,1) is Standard Normal Distribution
            else:
                z = torch.zeros_like(mu)
                negative_t_mask = t > 0
                z[negative_t_mask] = torch.randn_like(std[negative_t_mask])

            return mu + std * z # x_{t-1}

        # Denoised image at timestamp t-1
        pred_img = reparameterize(mu_tilde_t, var_t, t) # x_{t-1}; denoised image at timestep t-1

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return pred_img, x_hat_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))

    def get_time_pairs(self, times):
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        ##################################################################
        # TODO 3.2: Compute the output image for a single step of the DDIM
        # sampling process.
        ##################################################################
        tau_i, tau_isub1 = torch.tensor(tau_i, device=device).reshape([1]), torch.tensor(tau_isub1, device=device).reshape([1])
        
        # Step 1: Predicted noise & Predicted starting image at timestamp tau_i
        eps_tau_i, x_hat_0 = model_predictions(img, tau_i)

        if tau_isub1 >= 0:

            # Step 2: Extract alpha_tau_{i - 1} and alpha_tau_i
            alphas_tau_isub1 = extract(alphas_cumprod, tau_isub1, img.shape) # alpha_tau_{i - 1}
            alphas_tau_i = extract(alphas_cumprod, tau_i, img.shape) # alpha_tau_i

            # Step 3: Compute sigma_tau_i
            beta_tau_isub1 = 1 - alphas_tau_isub1 # beta_tau_{i - 1}
            # alphas_bar_tau_isub1 = alphas_cumprod[tau_isub1] # alpha_bar_tau_{i - 1}
            # alphas_bar_tau_i = alphas_cumprod[tau_i] # alpha_bar_tau_i
            alphas_bar_tau_isub1 = alphas_tau_isub1
            alphas_bar_tau_i = alphas_tau_i
            beta_tilde_tau_i = (1 - alphas_bar_tau_isub1) * beta_tau_isub1 / (1 - alphas_bar_tau_i) # beta_tilde_tau_i

            var_tau_i = eta * beta_tilde_tau_i # variance of Posterior Conditional Distribution at timestamp tau_i
            sigma_tau_i = torch.sqrt(var_tau_i) # standard dev of Posterior Conditional Distribution at timestamp tau_i

            # Step 4: Compute the coefficient of epsilon_tau_i
            eps_tau_i_coef = torch.sqrt(1 - alphas_bar_tau_isub1 - var_tau_i)

            # Step 5: Sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
            # HINT: Use the reparameterization trick
            mu_tilde_tau_i = torch.sqrt(alphas_bar_tau_isub1) * x_hat_0 + eps_tau_i_coef * eps_tau_i

            # Reparameterization Trick
            z = torch.randn_like(mu_tilde_tau_i)
            img = mu_tilde_tau_i + sigma_tau_i * z
        
        else:
            img = x_hat_0

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return img, x_hat_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)
        
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        return sample_fn(shape, z)
