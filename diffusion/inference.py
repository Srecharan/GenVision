import argparse
import os
import torch
import yaml
from torchvision.utils import save_image
from cleanfid import fid as cleanfid
from tqdm import tqdm

from models.unet import Unet
from models.diffusion import DDPM, DDIM

class DiffusionInference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.sample_dir = config['paths']['sample_dir']
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize UNet model
        self.model = Unet(
            dim=config['unet']['dim'],
            dim_mults=config['unet']['dim_mults'],
            channels=config['model']['channels']
        ).to(self.device)
        
        # Initialize diffusion model based on method
        if config['sampling']['method'] == 'ddpm':
            self.diffusion = DDPM(
                model=self.model,
                timesteps=config['model']['timesteps']
            ).to(self.device)
        else:
            self.diffusion = DDIM(
                model=self.model,
                timesteps=config['model']['timesteps'],
                sampling_timesteps=config['sampling']['ddim_timesteps'],
                ddim_sampling_eta=config['sampling']['ddim_eta']
            ).to(self.device)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('iteration', 0)

    @torch.no_grad()
    def generate_samples(self, num_samples=64):
        """Generate samples from the model."""
        self.model.eval()
        
        shape = (num_samples, self.config['model']['channels'], 
                self.config['model']['image_size'], 
                self.config['model']['image_size'])
        
        samples = self.diffusion.sample(shape)
        return samples

    @torch.no_grad()
    def compute_fid(self):
        """Compute FID score against CIFAR-10 dataset."""
        def sampling_function(z):
            shape = (-1, self.config['model']['channels'],
                    self.config['model']['image_size'],
                    self.config['model']['image_size'])
            samples = self.diffusion.sample(shape)
            return samples * 255  # Scale to [0, 255] range

        score = cleanfid.compute_fid(
            gen=sampling_function,
            dataset_name="cifar10",
            dataset_res=self.config['model']['image_size'],
            num_gen=self.config['sampling']['num_samples'],
            dataset_split="train",
            batch_size=self.config['sampling']['batch_size'],
            verbose=True
        )
        return score

def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=64,
                      help="Number of samples to generate")
    parser.add_argument("--compute_fid", action="store_true",
                      help="Compute FID score")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config with command line arguments
    config['sampling']['num_samples'] = args.num_samples

    # Initialize inference
    inference = DiffusionInference(config)
    inference.load_checkpoint(args.checkpoint)

    # Generate samples
    samples = inference.generate_samples(args.num_samples)
    save_image(
        samples,
        os.path.join(inference.sample_dir, 
                    f'samples_{config["sampling"]["method"]}.png'),
        nrow=8
    )

    # Compute FID if requested
    if args.compute_fid:
        fid_score = inference.compute_fid()
        print(f"FID Score: {fid_score:.2f}")

if __name__ == "__main__":
    main()