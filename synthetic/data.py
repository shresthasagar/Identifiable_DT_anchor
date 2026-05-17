import torch
import PIL
import os
import numpy as np
import scipy.io as sio
from scipy.stats import wishart
from numpy.random import uniform, choice
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, FashionMNIST
from torch.distributions.multivariate_normal import MultivariateNormal

import autoencoder


def gen_ssc(m, d, rho_coeff=1, maxnorm=1):
    rho = rho_coeff * np.sqrt(d)    # Set rho according to (Tatli et al., 2020)
    assert m > d, f"Must be m > d"
    assert rho > 1, f"Must be rho > 1"
    W = np.random.normal(0, rho**2/d, (m, d))
    for i in range(m):
        if np.linalg.norm(W[i]) > rho:
            W[i] *= rho / np.linalg.norm(W[i])
    W = np.sign(W) * np.minimum(np.abs(W), maxnorm)
    return W

class SynthDataset(Dataset):
    def __init__(self, X, S, transform):
        self.obs = X
        self.factors = S
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        x = self.obs[idx]
        if self.transform != None:
            x = self.transform(x)
        factors = self.factors[idx]
        return x, factors

class MNISTDataset(Dataset):
    def __init__(self, train):
        # Create data directory if it doesn't exist
        root = '/nfs/stak/users/nguyhoa3/scratch/data'
        os.makedirs(root, exist_ok=True)
        
        # Define transforms: resize to 32x32 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        # Download MNIST if not exists
        self.mnist = MNIST(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        return image
    
class FashionMNISTDataset(Dataset):
    def __init__(self, train):
        # Create data directory if it doesn't exist
        root = '/nfs/stak/users/nguyhoa3/scratch/data'
        # os.makedirs(root, exist_ok=True)
        
        # Define transforms: resize to 32x32 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        # Download FashionMNIST if not exists
        self.fashionmnist = FashionMNIST(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.fashionmnist)

    def __getitem__(self, idx):
        image, _ = self.fashionmnist[idx]
        return image
    
class EMNISTDataset(Dataset):
    def __init__(self, train):
        # Create data directory if it doesn't exist
        root = '/nfs/stak/users/nguyhoa3/scratch/data'
        os.makedirs(root, exist_ok=True)
        
        # Define transforms: resize to 32x32 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        # Download MNIST if not exists
        self.emnist = EMNIST(root=root, train=train, split='digits', download=True, transform=transform)

    def __len__(self):
        return len(self.emnist)

    def __getitem__(self, idx):
        image, _ = self.emnist[idx]
        return image

class CARS3DDataset():
    '''
    Cars3D dataset from "Deep Visual Analogy-Making" (Reed et al., 2015)
    183 car models, each with 24 rotation angles and 4 elevation levels
    Available at http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
    '''
    def __init__(self, dir='/nfs/stak/users/nguyhoa3/scratch/data/CARS3D/'):
        print("Loading CARS3D dataset...")
        for f in os.listdir(dir):
            if f == "cars3d.pt":
                self.cars3d = torch.load(os.path.join(dir, f))
                print(f"Loaded {self.__len__()} imgs.")
                return
        tmp = []
        for f in os.scandir(dir):
            if not f.is_file() or not f.name.endswith('.mat'):
                continue
            imgs = sio.loadmat(f)['im']
            tt = np.zeros((imgs.shape[3], imgs.shape[4], 64, 64, 3))
            for i in range(imgs.shape[3]):
                for j in range(imgs.shape[4]):
                    pic = PIL.Image.fromarray(imgs[:,:,:,i,j])
                    # Reshape images to 64x64x3
                    pic = pic.resize((64, 64), PIL.Image.LANCZOS)
                    tt[i, j, :, :, :] = np.array(pic)/255.0
            tmp.append(torch.tensor(tt.astype(np.float32)).permute(0,1,4,2,3))
            
        self.cars3d = torch.stack(tmp, dim=0)
        print(f"Loaded {len(tmp)}/183 car models.")

        # Flatten car model, rotation, and elevation dim
        self.cars3d = self.cars3d.view(-1, 3, 64, 64)
        print(f"Flattened to {self.__len__()} images, of dimension {self.cars3d.size()}.")

        # Save to disk
        torch.save(self.cars3d, os.path.join(dir, 'cars3d.pt'))

        return

    def __len__(self):
        return self.cars3d.size(0)

    def __getitem__(self, idx):
        image = self.cars3d[idx]
        return image

def get_real_data(args, num_workers=0):
    # Generate or load data depending on experiment
    if args.exp == 'mnist':
        train_data = MNISTDataset(train=True)
        val_data = MNISTDataset(args, train=False)
    elif args.exp == 'emnist':
        train_data = EMNISTDataset(train=True)
        val_data = EMNISTDataset(train=False)
    elif args.exp == 'cars3d':
        train_data = CARS3DDataset()
        val_data = CARS3DDataset()
    elif args.exp == 'fashionmnist':
        train_data = FashionMNISTDataset(train=True)
        val_data = FashionMNISTDataset(train=False)
    else:
        raise ValueError(f"Data {args.exp} not supported")

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.val_batchsize,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, val_loader

def get_synth_data(args, num_workers=0):
    # Sample covariance matrix
    sigma = wishart.rvs(args.latent_dim, np.eye(args.latent_dim), size=1)

    if args.sdist == 'gaussian':
        # Sample latents from Gaussian
        dist = MultivariateNormal(
            torch.FloatTensor([0.0] * args.latent_dim),
            torch.FloatTensor(sigma)
        )
        S = dist.sample([args.nobs]).float().numpy().reshape(args.nobs, args.latent_dim)
    elif args.sdist == 'uniform':
        # Sample latents from Uniform
        S = np.random.uniform(-1, 1, (args.nobs, args.latent_dim))
    elif args.sdist == 'uniform_dependent':
        # Sample latents from Uniform
        S1 = np.random.uniform(-1, 1, (args.nobs, 1))
        S2 = np.random.uniform(-1, 1, (args.nobs, 1)) + S1*0.5
        S = np.concatenate([S1, S2], axis=1)


    # Mix latents
    if args.exp == 'pnl':
        A = gen_ssc(args.m, args.latent_dim)
        # X = np.random.uniform(0.3, 0.5)*np.cos(S @ A.T) + S @ A.T
        X = 3*np.cos(S @ A.T) + S @ A.T

    
    elif args.exp == 'sparse_pnl':
        # set A to be a 2x2 permuation matrix 
        A = np.array([[0, 1], [1, 0]])
        X = np.random.uniform(0.3, 0.5)*np.cos(S @ A.T) + S @ A.T
        # X = 0.8*np.cos(S @ A.T) + S @ A.T


    elif args.exp == 'mlp':
        sparse_obs = np.random.binomial(1, args.sp_ratio, args.m)
        X = np.full((args.nobs, args.m), np.nan)
        for i in range(args.m):
            mixer = autoencoder.MLP(
                input_dim=args.latent_dim,
                output_dim=1,
                hidden_dim=8,
                n_layers=1,
                nonlinear='smooth_leaky_relu'
            )
            input_S = S.copy()
            
            # Almost sparsify
            if sparse_obs[i] == 1:
                alpha = uniform(args.low_alpha, args.high_alpha)
                sign = choice([-1, 1], size=1)
                which_s = choice(range(args.latent_dim), size=1)
                input_S[:, which_s] *= sign * alpha
            X[:, i] = mixer(torch.tensor(S, dtype=torch.float32)).detach().numpy().reshape(-1)

    X = X.astype(np.float32)
    S = S.astype(np.float32)

    # Train and validation splits
    S_train, S_val = np.split(S, [int(0.9 * len(S))])
    X_train, X_val = np.split(X, [int(0.9 * len(S))])

    # Create dataloaders
    train_loader = DataLoader(
        SynthDataset(X_train, S_train, transform=None),
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        SynthDataset(X_val, S_val, transform=None),
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader
