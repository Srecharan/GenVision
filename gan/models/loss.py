import torch
import torch.nn as nn

class GANLoss:
    """
    Standard GAN loss using binary cross entropy
    """
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_generator_loss(self, discrim_fake):
        valid_labels = torch.ones(discrim_fake.shape[0], 1, device=discrim_fake.device)
        return self.criterion(discrim_fake, valid_labels)

    def compute_discriminator_loss(self, discrim_real, discrim_fake):
        batch_size = discrim_real.shape[0]
        valid_labels = torch.ones(batch_size, 1, device=discrim_real.device)
        fake_labels = torch.zeros(batch_size, 1, device=discrim_fake.device)
        
        real_loss = self.criterion(discrim_real, valid_labels)
        fake_loss = self.criterion(discrim_fake, fake_labels)
        
        return real_loss + fake_loss

class WGANGPLoss:
    """
    Wasserstein GAN loss with gradient penalty
    """
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp

    def compute_generator_loss(self, discrim_fake):
        return -torch.mean(discrim_fake)

    def compute_discriminator_loss(self, discrim_real, discrim_fake, discrim_interp, interp):
        # Wasserstein loss
        loss_pt1 = torch.mean(discrim_fake) - torch.mean(discrim_real)
        
        # Gradient penalty
        gradients = torch.autograd.grad(
            outputs=discrim_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(discrim_interp),
            create_graph=True,
            retain_graph=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return loss_pt1 + gradient_penalty

class LSGANLoss:
    """
    Least Squares GAN loss
    """
    def compute_generator_loss(self, discrim_fake):
        return 0.5 * torch.mean((discrim_fake - 1) ** 2)

    def compute_discriminator_loss(self, discrim_real, discrim_fake):
        real_loss = 0.5 * torch.mean((discrim_real - 1) ** 2)
        fake_loss = 0.5 * torch.mean(discrim_fake ** 2)
        return real_loss + fake_loss