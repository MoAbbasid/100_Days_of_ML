<img width="995" alt="image" src="https://github.com/user-attachments/assets/8e00afe3-87be-4b8e-96c4-21ec6415df5e" />

# Frequentist & Bayesian Autoencoders

This document Contrasts frequentist (standard autoencoder) and Bayesian (variational autoencoder) approaches.
It explains how the reparameterization trick "moves the noise outside" the differentiable computation and describes the dual nature of the Evidence Lower Bound (ELBO).

---

## 1. Frequentist vs. Bayesian Approaches

### Frequentist (Standard Autoencoders)
- **Deterministic Encoding:**  
  - The encoder maps an input _x_ to a fixed latent vector.
  - Weight optimization is done via methods like Stochastic Gradient Descent (SGD) on fixed point estimates.
- **Sampling Role:**  
  - Randomness comes only from mini-batch selection.
  - The network's function remains fully differentiable.
- **Latent Representation:**  
  - The bottleneck produces a single compressed vector, which does not model uncertainty.

### Bayesian (Variational Autoencoders, VAEs)

- **Probabilistic Encoding:**  
  - The encoder outputs parameters (mean $\mu$ and variance $\sigma^2$) that define a distribution $q(z|x)$ (commonly Gaussian) over the latent variable.
  - This captures uncertainty in the representation.
- **Intrinsic Stochasticity:**  
  - The latent code is a distribution rather than a fixed point, requiring integration (or approximation) over possible values.
- **Latent Regularization:**  
  - A KL divergence term forces $q(z|x)$ to be close to a simple prior (typically N(0, I)).
  - This results in a smooth, continuous latent space useful for generation and interpolation.

---

## 2. Standard Autoencoder vs. Variational Autoencoder

| Aspect                    | Standard Autoencoder (Frequentist)             | Variational Autoencoder (Bayesian)                           |
|---------------------------|------------------------------------------------|--------------------------------------------------------------|
| **Latent Code**           | Fixed vector (point estimate)                  | Distribution (e.g., Gaussian with parameters μ and σ²)       |
| **Optimization**          | Minimizes reconstruction loss (deterministic)  | Maximizes the ELBO: both high reconstruction likelihood **and** low KL divergence  |
| **Uncertainty Modeling**  | Does not model uncertainty                     | Explicitly models uncertainty via sampling from $q(z|x)$    |
| **Sampling**              | Randomness only in mini-batch selection        | Randomness is inside the network (sampling from latent distributions) |

---

## 3. The Reparameterization Trick

### The Challenge:
- In VAEs, the encoder must sample _z_ from a distribution $q(z|x) = \mathcal{N}(\mu, \sigma^2)$.
- **Direct sampling is non-differentiable:**  
  - The stochastic sampling operation prevents gradients from flowing back to the parameters $\mu$ and $\sigma$.

### The Solution:
- **Reparameterize the Sampling:**  
  - Express _z_ as a deterministic function of the parameters and an auxiliary noise variable:
    $$z = \mu + \sigma \cdot \epsilon,\quad \text{with} \quad \epsilon \sim \mathcal{N}(0,1)$$
- **Key Idea:**  
  - The randomness is isolated in $\epsilon$ (which is independent of the network parameters), while the transformation involving $\mu$ and $\sigma$ is differentiable.
  - This enables gradients to backpropagate through $\mu$ and $\sigma$.

### Python Example:
```python
import torch

# Suppose the encoder outputs:
mu = torch.tensor([0.0])
log_var = torch.tensor([0.0])  # Represents log(σ²)
sigma = torch.exp(0.5 * log_var)

# Sample noise from standard normal distribution
epsilon = torch.randn_like(sigma)

# Reparameterized sample: noise is separated from μ and σ
z = mu + sigma * epsilon
```

---

## 4. Evidence Lower Bound (ELBO): A Dual Objective

### ELBO Definition:
$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$$
- **Reconstruction Term:**  
  - $\mathbb{E}_{q(z|x)}[\log p(x|z)]$
  - **Maximization:** Encourages the decoder to reconstruct the input accurately.
- **KL Divergence Term:**  
  - $D_{\text{KL}}\Big(q(z|x) \,\|\, p(z)\Big)$
  - **Minimization:** Regularizes the latent space by ensuring $q(z|x)$ remains close to a simple prior (typically $\mathcal{N}(0,I)$).

### Dual Nature:
- **Maximization:**  
  - Improve reconstruction quality by maximizing the expected log-likelihood.
- **Minimization:**  
  - Ensure a smooth, well-structured latent space by minimizing the divergence between the approximate posterior and the prior.
- This balance of objectives is central to training VAEs.

---

## 5. Summary

- **Standard Autoencoders (Frequentist):**
  - Provide a fixed, deterministic latent representation.
  - Use SGD and similar methods for weight updates.
  - Do not capture uncertainty in the latent code.

- **Variational Autoencoders (Bayesian):**
  - Represent the latent code as a probability distribution, capturing uncertainty.
  - Optimize a dual objective (ELBO) that both improves reconstruction and regularizes the latent space.
  - Use the reparameterization trick to allow gradient-based optimization through stochastic sampling.

- **Reparameterization Trick:**
  - Converts a non-differentiable sampling operation into a differentiable one by isolating the noise.
  - Enables standard backpropagation by "moving" the noise outside the core parameter-dependent computation.

- **ELBO Optimization:**
  - Balances two goals: accurate data reconstruction and maintaining a regular, semantically meaningful latent space.
  - Achieves this by simultaneously maximizing the reconstruction term and minimizing the KL divergence term.

---

- Next **β-VAE**?
- The β-VAE (beta-VAE) is a modification of the standard VAE that introduces a single hyperparameter β to better control the learning of disentangled representations.
- The regular VAE tries to balance two things:

- Reconstructing the input data well
- Making the latent space "nice" (through the KL divergence term)


 The β-VAE modifies this balance by introducing a β parameter:
```
CopyLoss = Reconstruction_Loss + β * KL_Divergence
```
- When β = 1, it's just a regular VAE
- When β > 1, it puts more emphasis on the KL divergence term, which encourages more disentanglement
- When β < 1, it focuses more on reconstruction quality
