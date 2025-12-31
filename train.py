import argparse
import os, sys
from dotenv import load_dotenv
import itertools
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torchvision.transforms.v2 as TT
import torch.nn.functional as F
# local imports
from networks.gan import Generator, Discriminator
from networks.vae import VAE
from networks.segformer import PartitionedSegformer
from utils.utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal, logits_to_labels
from utils.datasets import ImageDataset

# # TODO: fix BeyondClearPaths packaging and move to adding to a virtual environment with pip later
# # Load environment variables from 'imports.env'
# load_dotenv('imports.env')
# # Add the specified PYTHONPATH to sys.path
# python_path = os.getenv('PYTHONPATH')
# if python_path and python_path not in sys.path:
#     sys.path.append(python_path)
# from src.config import settings as BCP_settings
# from src import inference as BCP_infer

def load_model_checkpoint(checkpoint_path, model_cls = PartitionedSegformer.from_pretrained, device="cuda", model_kwargs=None):
    """ Load a pre-trained model checkpoint, assuming you have multiple checkpoint options, and we want to apply OOD detection on each
        Args:
            model_class: The model architecture
            checkpoint_path: Path to the checkpoint file
        Returns:
            The model loaded with the checkpoint weights
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    #print(model_kwargs)
    # kind of pointless but I have to do this unless I want to import a lot more of `transformers`'s Segformer default configurations
    pretrained_weights = model_kwargs.pop("pretrained_model_name_or_path")
    #print(model_kwargs)
    model = model_cls(pretrained_weights, **model_kwargs)  # Move model to GPU if available
    checkpoint = torch.load(checkpoint_path)
    state_dict_copy = {}
    # print("pretrained model keys: ", model.state_dict().keys())
    # print("\nmodel checkpoint keys: ", checkpoint["model_state_dict"].keys())
    for key, value in checkpoint["model_state_dict"].items():
        if "module." in key:
            state_dict_copy[key.replace("module.", "")] = value
        else:
            state_dict_copy[key] = value
    # print(checkpoint["model_state_dict"])
    #print("\nset difference of model keys and checkpoint keys: ", set(model.state_dict().keys()) - set(state_dict_copy.keys()))
    model.load_state_dict(state_dict_copy)
    model = model.to(device=device)
    return model



class CycleGANTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda' if torch.cuda.is_available() and opt.cuda else 'cpu'
        print(f"Using device: {self.device}")
        # Initialize networks
        self.netG_A2B = Generator(opt.input_nc, opt.output_nc).to(self.device)
        self.netG_B2A = Generator(opt.output_nc, opt.input_nc).to(self.device)
        self.netD_A = Discriminator(opt.input_nc).to(self.device)
        self.netD_B = Discriminator(opt.output_nc).to(self.device)
        self._initialize_weights()
        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=opt.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        # Learning rate schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        # Buffers and inputs
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        # Dataset loader
        self.dataloader = self._create_dataloader()
        self.logger = Logger(opt.n_epochs, len(self.dataloader))

    def _initialize_weights(self):
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

    def _create_dataloader(self):
        transforms_ = [
            TT.Resize((self.opt.size, self.opt.size), Image.BICUBIC),
            #TT.RandomCrop(self.opt.size),
            TT.RandomHorizontalFlip(p=0.1),
            TT.ToDtype(torch.float32, scale=True),
            #TT.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        dataset = ImageDataset(self.opt.dataroot, mode="train", transform=transforms_)
        return DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.n_cpu)


    def train(self):
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, batch in tqdm(enumerate(self.dataloader), desc=f"Epoch {epoch}"):
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                # Train Generators
                loss_G = self._train_generators(real_A, real_B)
                # Train Discriminators
                loss_D_A = self._train_discriminator(self.netD_A, real_A, self.fake_A_buffer, 'A')
                loss_D_B = self._train_discriminator(self.netD_B, real_B, self.fake_B_buffer, 'B')
                # Log progress
                images_to_log = {'fake_A': self.netG_B2A(real_B), 'fake_B': self.netG_A2B(real_A)} if epoch > 10 and i % 100 == 0 else None
                self.logger.log(
                    {
                        'loss_G': loss_G,
                        'loss_D_A': loss_D_A,
                        'loss_D_B': loss_D_B
                    },
                    images=images_to_log
                )
            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            # Save model checkpoints
            self.save_models(epoch)

    def _train_generators(self, real_A, real_B):
        # TODO: integrate masking for the generator in this function
        self.optimizer_G.zero_grad()
        # Identity loss
        loss_identity_A = self.criterion_identity(self.netG_B2A(real_A), real_A) * 5.0
        loss_identity_B = self.criterion_identity(self.netG_A2B(real_B), real_B) * 5.0
        # GAN loss
        fake_B = self.netG_A2B(real_A)
        loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), torch.ones_like(fake_B, device=self.device))
        fake_A = self.netG_B2A(real_B)
        loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), torch.ones_like(fake_A, device=self.device))
        # Cycle loss
        loss_cycle_ABA = self.criterion_cycle(self.netG_B2A(fake_B), real_A) * 10.0
        loss_cycle_BAB = self.criterion_cycle(self.netG_A2B(fake_A), real_B) * 10.0
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()

    def _train_discriminator(self, netD, real, buffer, domain):
        optimizer_D = self.optimizer_D_A if domain == 'A' else self.optimizer_D_B
        optimizer_D.zero_grad()
        # Real loss
        pred_real = netD(real)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real, device=self.device))
        # Fake Loss
        fake = buffer.push_and_pop(self.netG_B2A(real).detach()) if domain == 'A' else buffer.push_and_pop(self.netG_A2B(real).detach())
        pred_fake = netD(fake)
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()
        return loss_D.item()

    def save_models(self, epoch):
        torch.save(self.netG_A2B.state_dict(), f'output/netG_A2B_{epoch}.pth')
        torch.save(self.netG_B2A.state_dict(), f'output/netG_B2A_{epoch}.pth')
        torch.save(self.netD_A.state_dict(), f'output/netD_A_{epoch}.pth')
        torch.save(self.netD_B.state_dict(), f'output/netD_B_{epoch}.pth')



class DirtyGANTrainer(CycleGANTrainer):
    def __init__(self, opt, seg_model_kwargs):
        super().__init__(opt)
        # Add VAE for soiling pattern generation
        self.vae = VAE(input_nc=opt.input_nc, latent_dim=256).to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.criterion_vae = torch.nn.MSELoss()
        # Use pre-trained segmentation network (Segformer) for generating masks
        if "rgb_map" in seg_model_kwargs:
            self.rgb_map = seg_model_kwargs.pop("rgb_map")
        self.seg_network = self.load_segmentation_network(opt.segmentation_ckpt, seg_model_kwargs)
        # Update loss functions to include masked cycle consistency
        self.criterion_masked_cycle = torch.nn.L1Loss()
        self.mask_weight = opt.mask_weight if hasattr(opt, 'mask_weight') else 10.0
        # Preallocate tensors on the device for efficiency
        self.ones = torch.ones((opt.batch_size, 1), device=self.device)
        self.zeros = torch.zeros((opt.batch_size, 1), device=self.device)

    def load_segmentation_network(self, ckpt_path, model_kwargs={}):
        """Load your pre-trained segmentation network here."""
        # This function should load the segmentation network. Assuming it's already pre-trained.
        # For demonstration, returning a placeholder
        model = load_model_checkpoint(ckpt_path, **model_kwargs)
        model.eval()
        #return torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).eval()
        return model

    def _train_generators(self, real_A, real_B):
        self.optimizer_G.zero_grad()
        # identity loss from original CycleGAN
        soiled_to_clean = self.netG_B2A(real_B)
        # print("soiled_to_clean shape: ", soiled_to_clean.shape)
        # print("real A shape: ", real_A.shape)
        loss_identity_A = self.criterion_identity(soiled_to_clean, real_A) * 5.0
        clean_to_soiled = self.netG_A2B(real_A)
        # print("clean_to_soiled shape: ", clean_to_soiled.shape)
        # print("real B shape: ", real_B.shape)
        loss_identity_B = self.criterion_identity(clean_to_soiled, real_B) * 5.0
        # VAE soiling pattern generation
        soiling_pattern, _, _ = self.vae(real_A)
        # print("soiling pattern shape: ", soiling_pattern.shape)
        # generate Mask Using Segmentation Network
        #mask = self.generate_mask(soiling_pattern)
        mask = self.generate_mask(real_A)
        # print("mask shape: ", mask.shape)
        # apply VAE-generated soiling pattern using mask with linear combination
            #~ the interpolation step could probably be extended to be much more robust than this
        masked_real_A = real_A*(1 - mask) + soiling_pattern*mask
        # GAN Loss (Clean-to-Soiled)
        # print("masked real A shape: ", masked_real_A.shape)
        fake_B = self.netG_A2B(masked_real_A, mask)
        # print("fake_B shape: ", fake_B.shape)
        loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), self.ones[:fake_B.size(0)])
        # print("loss_GAN_A2B: ", loss_GAN_A2B)
        # GAN Loss (Soiled-to-Clean)
        fake_A = self.netG_B2A(real_B, None)
        # print("fake_A shape: ", fake_A.shape)
        loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), self.ones[:fake_A.size(0)])
        # print("loss_GAN_B2A: ", loss_GAN_B2A)
        # masked cycle consistency loss
        recov_A = self.netG_B2A(fake_B, mask)
        # print("recov_A", recov_A.shape)
        loss_cycle_ABA = self.criterion_masked_cycle(recov_A * mask, real_A * mask) * self.mask_weight
        # print("loss_cycle_ABA: ", loss_cycle_ABA)
        recov_B = self.netG_A2B(fake_A, None)
        # print("recov_B shape: ", recov_B.shape)
        loss_cycle_BAB = self.criterion_masked_cycle(recov_B, real_B) * 10.0
        # print("loss_cycle_BAB: ", loss_cycle_BAB)
        # VAE reconstruction loss using KL divergence
            #~ might actually try out using the WAE that does a reconstruction loss with maximum mean discrepancy
        recon_vae, mu, logvar = self.vae(real_A)
        # print("reconstructed vae shape: ", recon_vae.shape)
        # print("mu, logvar shapes: ", mu.shape, logvar.shape)
        kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
        # print("kl_div shape: ", kl_div.shape)
        loss_vae = self.criterion_vae(recon_vae, real_A) + kl_div
        # print("loss_vae: ", loss_vae)
        # total loss with backpropagation step
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_vae
        # print("loss_G: ", loss_G)
        loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_vae.step()
        return loss_G.item()

    def _train_discriminators(self, real_A, real_B):
        """Update discriminators for both clean and soiled domains."""
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()
        # Train D_A (Clean Domain)
        pred_real_A = self.netD_A(real_A)
        loss_real_A = self.criterion_GAN(pred_real_A, self.ones[:pred_real_A.size(0)])
        fake_A = self.fake_A_buffer.push_and_pop(self.netG_B2A(real_B).detach())
        pred_fake_A = self.netD_A(fake_A)
        loss_fake_A = self.criterion_GAN(pred_fake_A, self.zeros[:pred_fake_A.size(0)])
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        # Train D_B (Soiled Domain)
        pred_real_B = self.netD_B(real_B)
        loss_real_B = self.criterion_GAN(pred_real_B, self.ones[:pred_real_B.size(0)])
        fake_B = self.fake_B_buffer.push_and_pop(self.netG_A2B(real_A).detach())
        pred_fake_B = self.netD_B(fake_B)
        loss_fake_B = self.criterion_GAN(pred_fake_B, self.zeros[:pred_fake_B.size(0)])
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        loss_D = (loss_D_A + loss_D_B) * 0.5
        loss_D.backward()
        self.optimizer_D_A.step()
        self.optimizer_D_B.step()
        return loss_D.item()

    def generate_mask(self, image):
        """ Generate masks using the pre-trained segmentation network. """
        H, W = image.shape[-2:]
        num_classes = len(self.rgb_map)
        #? NOTE: if using segformer, there might be a need to collapse all soiled classes into one for binary masks - or create a new task for different levels of soiling altogether
        with torch.no_grad():
            logits = self.seg_network(image).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
            label_mask = logits_to_labels(logits, num_classes)
            # generate binary mask for pixels in any of the soiled classes (might have to rethink later)
            bin_mask = torch.zeros_like(label_mask, dtype=torch.bool)
            for i in range(2, num_classes):
                bin_mask = torch.logical_or(bin_mask, label_mask == i)
            return bin_mask.float()



if __name__ == "__main__":
    # vae = VAE(input_nc=3, latent_dim=256).to('cuda')
    # clean_to_soiled = torch.randn(4, 3, 256, 256).to('cuda')
    # output, mu, logvar = vae(clean_to_soiled)
    # print("Output shape:", output.shape)
    # print("Mu shape:", mu.shape)
    # print("Logvar shape:", logvar.shape)
    # sys.exit(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/')
    parser.add_argument('--segmentation_ckpt', type=str, default='checkpoints/segformer.pth')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay_epoch', type=int, default=60)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_cpu', type=int, default=4)
    opt = parser.parse_args()
    color_map = [
        {"label_idx": 0, "label": "clean", "rgb_tuple": (0, 0, 0)},
        {"label_idx": 1, "label": "transparent", "rgb_tuple": (0, 255, 0)},
        {"label_idx": 2, "label": "semi-transparent", "rgb_tuple": (0, 0, 255)},
        {"label_idx": 3, "label": "opaque", "rgb_tuple": (255, 0, 0)},
    ]
    model_kwargs = {
        # TODO: change to use the same initializations from the settings object in train_network.py
        # ~ While I'm at it, I may just add some of the functions to load model/training configurations currently in train_network.py to utils.py
        "pretrained_model_name_or_path": "nvidia/mit-b0",
        "num_labels": len(color_map),
        "id2label": {id: label_dict["label"] for id, label_dict in enumerate(color_map)},
        "label2id": {label_dict["label"]: id for id, label_dict in enumerate(color_map)}
    }
    if opt.dataroot is None:
        raise ValueError("Please specify the path to the dataset using the --dataroot flag")
    if opt.segmentation_ckpt is None:
        raise ValueError("Please specify the path to the segmentation checkpoint using the --segmentation_ckpt flag")
    # initialize training
    trainer = DirtyGANTrainer(opt, {"rgb_map": color_map, "model_kwargs": model_kwargs})
    trainer.train()