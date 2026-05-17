import torch
from torch import nn
from torch.nn import functional as F


# Smooth LeakyReLU for mixer
def smooth_leaky_relu(x, alpha=0.2):
    """
    Smoothen version of LeakyReLU
    Reference: https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function
    """
    return alpha * x + (1 - alpha) * torch.log(1 + torch.exp(x))
    
# Sparsity reg (minimax concave penalty)
def mcp(x, gamma=2, lam=0.5):
    assert gamma > 1, f"Gamma must be greater than 1, given {gamma}"

    condition1 = torch.abs(x) <= gamma * lam
    condition2 = torch.abs(x) > gamma * lam

    result = torch.zeros_like(x)
    result[condition1] = lam * torch.abs(x[condition1]) - (x[condition1] ** 2) / (2 * gamma)
    result[condition2] = 0.5 * gamma * (lam ** 2)

    return result

# Init weights
def init_weights(m):
    if isinstance(m, nn.Conv2d)\
        or isinstance(m, nn.ConvTranspose2d)\
            or isinstance(m, nn.Linear)\
                or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# Multi-layer perceptron
class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=32, n_layers=2, nonlinear='leaky_relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)

        if nonlinear == 'leaky_relu':
            self._act_f = lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif nonlinear == 'smooth_leaky_relu':
            self._act_f = lambda x: smooth_leaky_relu(x, alpha=0.2)

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f(self.fc[c](h))
        return h

# Encodes a 32x32 grayscale image into a latent representation of size [B, 10, 1, 1]
class MNISTEncoder(nn.Module):
    def __init__(self, d=10, img_dim=32, channels=1):
        super().__init__()
        assert img_dim == 32, "Image dimension must be 32"
        self.layers = nn.Sequential(
            # Layer 1: [B, 1, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(channels, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Layer 2: [B, 256, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Layer 3: [B, 128, 8, 8] -> [B, 64, 4, 4]
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Layer 4: [B, 64, 4, 4] -> [B, 32, 2, 2]
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Layer 5: [B, 32, 2, 2] -> [B, 10, 1, 1]
            nn.Conv2d(32, d, kernel_size=2)
        )

    def forward(self, x):
        return self.layers(x)

# Decodes a latent representation of size [B, latent_dim, 1, 1] back to a 32x32 grayscale image.
class MNISTDecoder(nn.Module):
    def __init__(self, d=10, img_dim=32, channels=1):
        super().__init__()
        assert img_dim == 32, "Image dimension must be 32"
        self.layers = nn.Sequential(
            # Layer 1: [B, d, 1, 1] -> [B, 32, 2, 2]
            nn.ConvTranspose2d(d, 32, kernel_size=2),
            nn.LeakyReLU(0.2),
            # Layer 2: [B, 32, 2, 2] -> [B, 64, 4, 4]
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            # Layer 3: [B, 64, 4, 4] -> [B, 128, 8, 8]
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            # Layer 4: [B, 128, 8, 8] -> [B, 256, 16, 16]
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            # Layer 5: [B, 256, 16, 16] -> [B, 1, 32, 32]
            nn.ConvTranspose2d(256, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, s):
        return self.layers(s)

class CARS3DEncoder(nn.Module):
    def __init__(self, latent_dim, img_dim=64, channels=3):
        super().__init__()
        assert img_dim == 64, "Image dimension must be 64"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(4*4*64, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        x = self.linear_layers(x)
        return x

class CARS3DDecoder(nn.Module):
    def __init__(self, latent_dim, img_dim=64, channels=3):
        super().__init__()
        assert img_dim == 64, "Image dimension must be 64"
        self.linear_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = nn.Unflatten(1, (64, 4, 4))(x) # [B, 64, 4, 4]
        x = self.deconv_layers(x)
        return x

# Encoder + Decoder
class AutoEncoder(torch.nn.Module):
    def __init__(self, d, encoder, decoder):
        super().__init__()
        self.d = d
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # encode
        sh = self.encoder(x)

        # decode
        xh = self.decoder(sh)

        # returns inferred latents and reconstructed observation
        return sh, xh
    
# Get model from args
def get_model(args):
    # Get model's device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    # Get encoders and decoders
    if args.exp == 'pnl' or args.exp == 'mlp' or args.exp == 'sparse_pnl':
        encoder = MLP(
            input_dim=args.m,
            output_dim=args.latent_dim,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            nonlinear='leaky_relu'
        ).to(device)
        decoder = MLP(
            input_dim=args.latent_dim,
            output_dim=args.m,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            nonlinear='leaky_relu'
        ).to(device)
    elif args.exp == 'mnist' or args.exp == 'emnist':
        encoder = MNISTEncoder(
            d=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)
        decoder = MNISTDecoder(
            d=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)
    elif args.exp == 'cars3d':
        encoder = CARS3DEncoder(
            latent_dim=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)
        decoder = CARS3DDecoder(
            latent_dim=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)
    elif args.exp == 'fashionmnist':
        encoder = MNISTEncoder(
            d=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)
        decoder = MNISTDecoder(
            d=args.latent_dim,
            img_dim=args.img_dim,
            channels=args.channels
        ).to(device)

    # Get autoencoder
    autoencoder = AutoEncoder(
        d=args.latent_dim,
        encoder=encoder,
        decoder=decoder,
    ).to(device)

    return autoencoder