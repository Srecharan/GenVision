# GenVision: Advanced Generative Models for Image Synthesis

## Overview

This project explores three fundamentally different approaches to deep generative modeling:

*   **Generative Adversarial Networks (GANs):** Implementation of three variants (Vanilla GAN, LSGAN, WGAN-GP) with progressive improvements in stability and sample quality.
*   **Variational Autoencoders (VAEs):** Exploration of different latent space dimensions and β-annealing strategies for optimal reconstruction-generation balance.
*   **Diffusion Models:** Implementation of efficient sampling procedures (DDPM and DDIM) with pre-trained models.

<div align="center">
  <a href="assets/GenVision.pdf">
    <img src="assets/first_page.png" width="600" alt="GenVision Technical Overview"/>
    <p><i>Click on the image to view the complete Technical Overview PDF</i></p>
  </a>
</div>

## Model Architecture Details

### GAN Implementation
- Generator: Progressive upsampling with ResBlocks
- Discriminator: Convolutional architecture with spectral normalization
- Custom loss functions for each GAN variant

<div align="center">
  <img src="assets/gan_figure.png" width="700"/>
  <p><i>GAN architecture with generator and discriminator components and loss function variants</i></p>
</div>

### VAE Components
- Encoder: Deep convolutional network with variable latent dimensions
- Decoder: Mirror architecture with transposed convolutions
- KL divergence scheduling with β-annealing

<div align="center">
  <img src="assets/vae_figure.png" width="700"/>
  <p><i>VAE architecture showing encoder, latent space, and decoder with loss function components</i></p>
</div>

### Diffusion Architecture
- U-Net backbone with skip connections
- Self-attention layers at multiple scales
- Cosine noise schedule implementation

<div align="center">
  <img src="assets/diffusion_figure.png" width="700"/>
  <p><i>Diffusion model architecture showing forward and reverse processes with sampling strategies</i></p>
</div>


### 1. GAN Results

| Model      | FID Score | Sample Quality | Training Stability |
|------------|-----------|----------------|-------------------|
| Vanilla GAN | 104.62    | Poor           | Unstable          |
| LSGAN      | 52.48     | Improved       | More Stable       |
| WGAN-GP    | 33.07     | Best           | Stable            |

#### Vanilla GAN
<table>
<tr>
<td><img src="assets/Vanilla_GAN_Samples.png" alt="Vanilla GAN Samples" width="400"/></td>
<td><img src="assets/Vanilla GAN Latent Space Interpolations.png" alt="Vanilla GAN Interpolations" width="400"/></td>
</tr>
<tr>
<td align="center">Samples</td>
<td align="center">Latent Space Interpolation</td>
</tr>
</table>

#### LSGAN
<table>
<tr>
<td><img src="assets/LS-GAN Samples.png" alt="LSGAN Samples" width="400"/></td>
<td><img src="assets/LS-GAN Latent Space Interpolations.png" alt="LSGAN Interpolations" width="400"/></td>
</tr>
<tr>
<td align="center">Samples</td>
<td align="center">Latent Space Interpolation</td>
</tr>
</table>

#### WGAN-GP
<table>
<tr>
<td><img src="assets/WGAN-GP Samples.png" alt="WGAN-GP Samples" width="400"/></td>
<td><img src="assets/WGAN-GP Latent Space Interpolations.png" alt="WGAN-GP Interpolations" width="400"/></td>
</tr>
<tr>
<td align="center">Samples</td>
<td align="center">Latent Space Interpolation</td>
</tr>
</table>

### 2. VAE Results

#### Performance Analysis

| Model | Latent Size | Reconstruction Loss | Sample Quality |
|-------|-------------|-------------------|----------------|
| AE    | 16          | ~200              | Poor           |
| AE    | 128         | ~75               | Good           |
| AE    | 1024        | ~40               | Best           |
| VAE (β=0.8) | 128    | ~125             | Improved       |

#### Autoencoder Loss Curves
<table>
<tr>
<td><img src="assets/Loss (size 16).png" alt="Loss Size 16" width="270"/></td>
<td><img src="assets/Loss (size 128).png" alt="Loss Size 128" width="270"/></td>
<td><img src="assets/Loss (size 1024).png" alt="Loss Size 1024" width="270"/></td>
</tr>
<tr>
<td align="center">Latent Size: 16</td>
<td align="center">Latent Size: 128</td>
<td align="center">Latent Size: 1024</td>
</tr>
</table>

#### Autoencoder Reconstructions
<table>
<tr>
<td><img src="assets/Reconstructions: (size 16).png" alt="Recon Size 16" width="270"/></td>
<td><img src="assets/Reconstructions: (size 128).png" alt="Recon Size 128" width="270"/></td>
<td><img src="assets/Reconstructions: (size 1024).png" alt="Recon Size 1024" width="270"/></td>
</tr>
<tr>
<td align="center">Latent Size: 16</td>
<td align="center">Latent Size: 128</td>
<td align="center">Latent Size: 1024</td>
</tr>
</table>

#### VAE with β-annealing Results
- Best β value: 0.8 (selected based on sample quality and reconstruction fidelity)
<table>
<tr>
<td><img src="assets/Recon. Loss: β annealed.png" alt="Recon Loss Beta" width="400"/></td>
<td><img src="assets/KL Loss: β annealed.png" alt="KL Loss Beta" width="400"/></td>
<td><img src="assets/Samples: β annealed.png" alt="Samples Beta" width="400"/></td>
</tr>
<tr>
<td align="center">Reconstruction Loss</td>
<td align="center">KL Loss</td>
<td align="center">Generated Samples</td>
</tr>
</table>

**Key Findings:**
- Larger latent sizes (1024) produce significantly better reconstructions
- β-annealing improves sample quality while maintaining good reconstruction
- Trade-off between reconstruction quality and sampling capability is evident across different configurations

### 3. Diffusion Model Results

| Model | FID Score | Sampling Speed |
|-------|-----------|---------------|
| DDPM  | 34.73     | Slower        |
| DDIM  | 38.32     | Faster        |

<table>
<tr>
<td><img src="assets/DDPM Samples.png" alt="DDPM Samples" width="400"/></td>
<td><img src="assets/DDIM Samples.png" alt="DDIM Samples" width="400"/></td>
</tr>
<tr>
<td align="center">DDPM Samples</td>
<td align="center">DDIM Samples</td>
</tr>
</table>

## References

- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [VAE Paper](https://arxiv.org/abs/1312.6114)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
