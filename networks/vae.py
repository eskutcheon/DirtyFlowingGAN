import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_nc=3, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute the size of the flattened output from the encoder
        sample_input = torch.randn(1, input_nc, 256, 256)
        with torch.no_grad():
            self.encoder_output_shape = self.encoder(sample_input).shape
        self.flat_dim = self.encoder_output_shape[1] * self.encoder_output_shape[2] * self.encoder_output_shape[3]
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        # self.fc_mu = nn.Linear(256 * 64 * 64, latent_dim)
        # self.fc_logvar = nn.Linear(256 * 64 * 64, latent_dim)
        # self.fc_decode = nn.Linear(latent_dim, 256 * 64 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        C, H, W = self.encoder_output_shape[-3:]
        z = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        # Decode
        z = self.fc_decode(z).view(x.size(0), C, H, W)
        #z = self.fc_decode(z).view(x.size(0), 256, 64, 64)
        return self.decoder(z), mu, logvar
