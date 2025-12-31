import argparse
import sys
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
# local imports
from networks.gan import Generator
from utils.datasets import ImageDataset


# TODO: update for the new dataset and generator/discriminator structure

class CycleGANTester:
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda' if torch.cuda.is_available() and opt.cuda else 'cpu'
        print(f"Using device: {self.device}")
        # Initialize models
        self.netG_A2B = Generator(opt.input_nc, opt.output_nc).to(self.device)
        self.netG_B2A = Generator(opt.output_nc, opt.input_nc).to(self.device)
        self._load_models()
        # Set models to evaluation mode
        self.netG_A2B.eval()
        self.netG_B2A.eval()
        # Tensor type based on device
        self.Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        # Create output directories if they don't exist
        self._create_output_dirs()
        # Prepare data loader
        self.dataloader = self._create_dataloader()

    def _load_models(self):
        """Load the pre-trained generator models."""
        self.netG_A2B.load_state_dict(torch.load(self.opt.generator_A2B, map_location=self.device))
        self.netG_B2A.load_state_dict(torch.load(self.opt.generator_B2A, map_location=self.device))
        print("Loaded pre-trained models.")

    def _create_output_dirs(self):
        """Create directories for saving output images."""
        os.makedirs('output/A', exist_ok=True)
        os.makedirs('output/B', exist_ok=True)

    def _create_dataloader(self):
        """Create a DataLoader for the test dataset."""
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        dataset = ImageDataset(self.opt.dataroot, transforms_=transforms_, mode='test')
        return DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=False, num_workers=self.opt.n_cpu)

    def test(self):
        """Run the testing pipeline."""
        for i, batch in enumerate(self.dataloader):
            real_A = Variable(self.Tensor(batch['A']).to(self.device))
            real_B = Variable(self.Tensor(batch['B']).to(self.device))
            # Generate fake images
            fake_B = self._generate_image(self.netG_A2B, real_A)
            fake_A = self._generate_image(self.netG_B2A, real_B)
            # Save generated images
            self._save_image(fake_A, 'A', i + 1)
            self._save_image(fake_B, 'B', i + 1)
            sys.stdout.write(f'\rGenerated images {i + 1} of {len(self.dataloader)}')
        sys.stdout.write('\n')

    def _generate_image(self, model, input_image):
        """Generate an output image using the specified model."""
        with torch.no_grad():
            fake_image = model(input_image)
            return 0.5 * (fake_image + 1.0)

    def _save_image(self, image, domain, index):
        """Save the generated image to the appropriate directory."""
        save_image(image, f'output/{domain}/{index:04d}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    # run inference
    tester = CycleGANTester(opt)
    tester.test()