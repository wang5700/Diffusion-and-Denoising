# Diffusion Model for MNIST Denoising

This repository contains a PyTorch implementation of a diffusion model designed to denoise MNIST images. The model is based on a time-conditioned U-Net architecture, trained to predict the reverse-step mean in a denoising diffusion process. The project includes code for network architecture, diffusion process, loss computation, training, sampling, and evaluation using the Fréchet Inception Distance (FID) score.

## Project Overview

The project implements a diffusion model for generating MNIST digits by learning to reverse a noise-adding process. Key components include:


1. Network Architecture: A U-Net model (ScoreNet_channel_reduc) that predicts the mean of the reverse diffusion step, conditioned on time using Gaussian Fourier embeddings.



2. Diffusion Process: A Diffusion class that manages the forward process (adding noise) and predicts the reverse mean using the trained model.



3. Loss Function: A Monte Carlo estimator for the loss, averaging over multiple timesteps to reduce variance.



4. Training: Code to train the model on the MNIST dataset with a learning rate scheduler.



5. Sampling: Implementation of the reverse diffusion process to generate samples (Eq. 5).



6. Evaluation: Computation of the FID score to evaluate generated samples against real MNIST data.

The project compares two loss formulations: the original $x_{t-1}$ loss  $\mathbb{E}\Bigl[\|\tilde\mu_t(X_t,X_0)-\mu(X_t,t;\theta)\|^2/(2(1-\alpha_t))\Bigr]$
 and the noise $\epsilon$ loss $\mathbb{E}\Bigl[\tfrac{1-\alpha_t}{2\alpha_t(1-\bar\alpha_t)}\|\epsilon_t - e_\theta(X_t,t)\|^2\Bigr]$ , highlighting their practical differences in training stability and performance.

