import random
import torch
import argparse
from eval import  plot_synth
import utils
import wandb
import os
import autoencoder
import data
from models import Discriminator  # Import the Discriminator class
from tqdm import tqdm
import math


def compute_te(model, data_loader, device, latent_dim):
    """
    Compute Translation Error (TE) over a data loader.
    
    TE = (1/sqrt(D)) * ||g_hat(x) - y||_2
    
    where x is the source sample, y is the target sample, and D is the latent dimension.
    
    Returns:
        te_mean: Mean TE over all samples
        te_std: Standard deviation of TE over all samples
    """
    model.eval()
    te_values = []
    
    with torch.no_grad():
        for x, s in data_loader:
            x = x.to(device)
            s = s.to(device)
            
            # Get model predictions (extracted sources)
            sh, _ = model(x)
            sh = sh.reshape(sh.shape[0], latent_dim)
            
            # Compute per-sample TE: (1/sqrt(D)) * ||sh - s||_2
            # ||sh - s||_2 for each sample (L2 norm along latent dimension)
            l2_norm = torch.norm(sh - s, p=2, dim=1)
            te = l2_norm / math.sqrt(latent_dim)
            
            te_values.append(te)
    
    # Concatenate all TE values
    all_te = torch.cat(te_values, dim=0)
    
    te_mean = all_te.mean().item()
    te_std = all_te.std().item()
    
    model.train()
    return te_mean, te_std


def train(args):
    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Fix random seed
    # seed = random.randint(0, 10000)
    
    # seed = 42
    seed = 24
    utils.set_seed(seed)

    # Create model
    model = autoencoder.get_model(args)
    
    # Create discriminator if GAN loss is enabled
    discriminator = None
    if args.lam_gan > 0:
        discriminator = Discriminator(input_dim=args.latent_dim, 
                                     hidden_dim=args.disc_hidden_dim, 
                                     n_layers=args.disc_n_layers).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load data
    train_loader, val_loader = data.get_synth_data(args)
    
    # Get true sources for GAN training
    true_sources = None
    if args.lam_gan > 0:
        # Get a batch of true sources from the data loader
        _, true_sources = next(iter(train_loader))
        true_sources = true_sources.to(device)
        
    # Initialize anchor points
    anchor_x = None
    anchor_s = None
    
    # Train loop
    b_it, glob_it = 0, 0 
    run_recon_loss = 0.0
    run_gen_loss = 0.0
    run_disc_loss = 0.0
    run_anchor_loss = 0.0
    
    pbar = tqdm(total=args.num_iters, desc="Training")
    while glob_it < args.num_iters:
        model.train()
        if discriminator is not None:
            discriminator.train()
            
        x, s = next(iter(train_loader))
        b_it += 1
        optimizer.zero_grad()
        x = x.to(device)
        s = s.to(device)
        
        # Save anchor points from the first batch
        if glob_it == 0 and args.lam_anchor > 0:
            # Select random indices for anchor points
            n_anchors = min(args.n_anchors, x.size(0))
            indices = torch.randperm(x.size(0))[:n_anchors]
            anchor_x = x[indices].clone().detach()
            anchor_s = s[indices].clone().detach()
            print(f"Saved {n_anchors} anchor points")

        sh, xh = model(x)
        sh = sh.reshape(sh.shape[0], args.latent_dim)
        total_loss = None

        # Recon loss
        recon_loss = (x - xh).square().mean()
        run_recon_loss += recon_loss.item()

        # Get Jacobian
        jacobian = torch.vmap(torch.func.jacfwd(model.decoder))(sh.flatten(1))

        # Volume reg
        if args.lam_vol > 0:
            volreg = torch.mean(torch.det(jacobian.transpose(-1, -2) @ jacobian))
        else:
            with torch.no_grad():
                volreg = torch.Tensor([0.0]).to(device)

        # Maxnorm reg
        if args.lam_maxnorm > 0:
            maxnormreg = torch.mean(torch.nn.functional.relu(torch.amax(torch.abs(jacobian), dim=(1, 2)) - 1))
        else:
            with torch.no_grad():
                maxnormreg = torch.Tensor([0.0]).to(device)

        # L1-norm reg
        if args.lam_l1norm > 0:
            l1normreg = torch.mean(torch.nn.functional.relu(torch.abs(jacobian).sum(dim=(1, 2)) - 1))
        else:
            with torch.no_grad():
                l1normreg = torch.Tensor([0.0]).to(device)

        # Sparse reg
        if args.lam_sp > 0:
            spreg = torch.mean(autoencoder.mcp(jacobian))
        else:
            with torch.no_grad():
                spreg = torch.Tensor([0.0]).to(device)

        # GAN loss
        gan_loss = torch.Tensor([0.0]).to(device)
        disc_loss_val = 0.0
        gen_loss_val = 0.0
        if args.lam_gan > 0 and discriminator is not None:
            # Train discriminator
            disc_optimizer.zero_grad()
            
            # Real sources - use the true sources from the data
            real_sources = s
            real_pred = discriminator(real_sources)
            real_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            
            # Fake sources (extracted)
            fake_pred = discriminator(sh.detach())  # Detach to avoid training generator
            fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            
            # Total discriminator loss
            disc_loss = real_loss + fake_loss
            disc_loss_val = disc_loss.item()
            disc_loss.backward()
            disc_optimizer.step()
            run_disc_loss += disc_loss_val
            
            # Train generator with non-saturating GAN loss
            gen_pred = discriminator(sh)
            # Non-saturating loss: maximize log(D(G(z))) instead of minimize log(1-D(G(z)))
            gan_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gen_pred, torch.ones_like(gen_pred)
            )
            gen_loss_val = gan_loss.item()
            run_gen_loss += gen_loss_val
            
        # Anchor point matching loss
        anchor_loss = torch.Tensor([0.0]).to(device)
        if args.lam_anchor > 0 and anchor_x is not None:
            # Get model predictions for anchor points
            anchor_sh, _ = model(anchor_x)
            anchor_sh = anchor_sh.reshape(anchor_sh.shape[0], args.latent_dim)
            
            # Calculate MSE loss between original and extracted sources
            anchor_loss = torch.nn.functional.mse_loss(anchor_sh, anchor_s)
            run_anchor_loss += anchor_loss.item()

        # Total loss
        total_loss = args.lam_recon * recon_loss \
                    - args.lam_vol * volreg  \
                    + args.lam_maxnorm * maxnormreg \
                    + args.lam_l1norm * l1normreg \
                    + args.lam_sp * spreg \
                    + args.lam_gan * gan_loss \
                    + args.lam_anchor * anchor_loss

        # Backprop and update
        total_loss.backward()
        optimizer.step()
        glob_it += 1
        pbar.update(1)

        # Log training losses
        if glob_it == 1 or glob_it % args.eval_iter == 0 or glob_it == args.num_iters:
            train_recon = run_recon_loss / b_it
            
            # Calculate average GAN losses
            train_gen_loss = run_gen_loss / b_it if args.lam_gan > 0 else 0.0
            train_disc_loss = run_disc_loss / b_it if args.lam_gan > 0 else 0.0
            train_anchor_loss = run_anchor_loss / b_it if args.lam_anchor > 0 else 0.0
            
            b_it = 0
            run_recon_loss = 0.0
            run_gen_loss = 0.0
            run_disc_loss = 0.0
            run_anchor_loss = 0.0

            # Log to W&B
            log_dict = {
                "train_recon": train_recon,
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
                "volreg": volreg.item(),
                "maxnormreg": maxnormreg.item(),
                "l1normreg": l1normreg.item(),
                "spreg": spreg.item()
            }
            
            # Add GAN losses to log if enabled
            if args.lam_gan > 0:
                log_dict.update({
                    "gan_loss": gan_loss.item(),
                    "train_gen_loss": train_gen_loss,
                    "train_disc_loss": train_disc_loss
                })
                
            # Add anchor loss to log if enabled
            if args.lam_anchor > 0:
                log_dict.update({
                    "anchor_loss": anchor_loss.item(),
                    "train_anchor_loss": train_anchor_loss
                })
            
            # Compute and log Translation Error (TE)
            te_mean, te_std = compute_te(model, val_loader, device, args.latent_dim)
            log_dict.update({
                "te_mean": te_mean,
                "te_std": te_std
            })
                
            wandb.log(log_dict)

        # Only generate and log plots at specified intervals
        if args.plot_iter > 0 and (glob_it == 1 or glob_it % args.plot_iter == 0 or glob_it == args.num_iters):
            fig = plot_synth(args, model, val_loader, filename=f"synth_plot_{args.run_name}.pkl")
            wandb.log({
                "synth_plot": wandb.Image(fig)
            })

    pbar.close()
    
    # Compute final Translation Error
    final_te_mean, final_te_std = compute_te(model, val_loader, device, args.latent_dim)
    print(f"Final TE: {final_te_mean:.4f} ± {final_te_std:.4f}")
    
    return final_te_mean, final_te_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers",
        type=int,
        help="No. of hidden layers in encoder and decoder",
        default=2,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        help="Hidden dim in encoder and decoder",
        default=32
    )
    parser.add_argument(
        "--lam_vol",
        help="Volume reg",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lam_recon",
        help="Reconstruction reg",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--lam_maxnorm",
        help="Maxnorm reg",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--lam_l1norm",
        help="L1-norm reg",
        type=float,
        default=1e-1
    )
    parser.add_argument(
        "--lam_sp",
        help="Sparse reg",
        type=float,
        default=0.0
    )
    # Add GAN-related arguments
    parser.add_argument(
        "--lam_gan",
        help="GAN loss weight",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--disc_n_layers",
        help="Number of hidden layers in discriminator",
        type=int,
        default=2
    )
    parser.add_argument(
        "--disc_hidden_dim",
        help="Hidden dimension in discriminator",
        type=int,
        default=64
    )
    parser.add_argument(
        "--disc_lr",
        help="Learning rate for discriminator",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default="1024"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default="1e-3"
    )
    parser.add_argument(
        "--num_iters",
        help="Number of training iter",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--eval_iter",
        help="Evaluating every no. of iterations given by arg",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--nobs",
        help="Number of samples",
        type=int,
        default=30000
    )
    parser.add_argument(
        "--latent_dim",
        help="Latent dim",
        type=int,
        default=2
    )
    parser.add_argument(
        "--m",
        help="Obs dim",
        type=int,
        default=64
    )
    parser.add_argument(
        "--cuda",
        help="CUDA device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--exp",
        help="Exp type (pnl, mlp)",
        type=str,
        default='mlp'
    )
    parser.add_argument(
        "--sp_ratio",
        help="Ratio of almost-sparsifed obs",
        type=str,
        default=0.5
    )
    parser.add_argument(
        "--low_alpha",
        help="Lower bound for alpha",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--high_alpha",
        help="Upper bound for alpha",
        type=float,
        default=0.0002
    )
    parser.add_argument(
        "--sdist",
        help="Latent dist (gaussian, uniform)",
        type=str,
        default='uniform'
    )
    parser.add_argument(
        "--lam_anchor",
        help="Anchor point matching loss weight",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--n_anchors",
        help="Number of anchor points to use",
        type=int,
        default=100
    )
    parser.add_argument(
        "--run_name",
        help="Custom name for the wandb run",
        type=str,
        default=None
    )
    parser.add_argument(
        "--plot_iter",
        help="Plot every n iterations (0 to disable plotting)",
        type=int,
        default=1000
    )

    args = parser.parse_args()
    
    # W&B init
    wandb.init(project=f"dica_synth_{args.exp}",
               config=vars(args),
               name=args.run_name,
               settings=wandb.Settings(_disable_stats=True)
    )

    # Train model
    te_mean, te_std = train(args)

    # W&B summary and finish
    if te_mean is not None:
        wandb.run.summary["final_te_mean"] = te_mean
    if te_std is not None:
        wandb.run.summary["final_te_std"] = te_std
    wandb.finish()
