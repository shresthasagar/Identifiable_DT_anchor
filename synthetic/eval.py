import numpy as np
import torch
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import kernel_ridge
import torch
import utils
from autoencoder import mcp


def mcc(Z: torch.Tensor, hZ: torch.Tensor) -> np.ndarray:
    d = Z.shape[1]
    hZ = hZ.cpu()
    Z = Z.cpu()
    
    # Compute Pearson correlation matrix
    mat = (torch.abs(torch.corrcoef(torch.cat((Z.T, hZ.T), 0))))[:d, d:]

    return mat

def r2(Z: torch.Tensor, hZ: torch.Tensor) -> np.ndarray:
    d = Z.shape[1]

    # Initialize matrix of R2 scores
    mat = np.zeros((d, d))
    hZ = hZ.cpu()
    Z = Z.cpu()

    # Use KRR to predict ground-truth from inferred latents
    reg_func = lambda: kernel_ridge.KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    for i in range(d):
        for j in range(d):
            ZS = Z[:, i].reshape(-1, 1)
            hZS = hZ[:, j].reshape(-1, 1)

            # Standardize latents (cuz Gaussian kernel)
            scaler_Z = StandardScaler()
            scaler_hZ = StandardScaler()

            z_train, z_eval = np.split(scaler_Z.fit_transform(ZS), [int(0.9 * len(ZS))])
            hz_train, hz_eval = np.split(
                scaler_hZ.fit_transform(hZS), [int(0.9 * len(hZS))]
            )

            # Fit KRR model
            reg_model = reg_func()
            reg_model.fit(hz_train, z_train)
            hz_pred_val = reg_model.predict(hz_eval)

            # Populate R2 score matrix
            mat[i, j] = sklearn.metrics.r2_score(z_eval, hz_pred_val)

    return mat

def eval_real(args, model, data_loader):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        run_recon_loss = 0
        run_vol = 0
        run_maxnorm = 0
        run_l1norm = 0
        run_sp = 0
        run_l0norm = 0
        b_it = 0
        for x in data_loader:
            x = x.to(device)
            sh, xh = model(x)

            # Get reconstruction loss
            run_recon_loss += ((x - xh).square().mean()).item()

            # Get regularizers
            val_jac = torch.vmap(torch.func.jacfwd(model.decoder))(sh)
            if args.vol_surr == "det":
                run_vol += torch.mean(torch.det(val_jac.transpose(-1, -2) @ val_jac))
            elif args.vol_surr == "logdet":
                if args.tau > 0:
                    volreg = torch.mean(torch.logdet(val_jac.transpose(-1, -2) @ val_jac + args.tau*torch.eye(args.latent_dim, device=device)))
                else:
                    volreg = torch.mean(torch.logdet(val_jac.transpose(-1, -2) @ val_jac))
            run_maxnorm += torch.mean(torch.amax(torch.abs(val_jac), dim=(1, 2)))
            run_l1norm += torch.mean(torch.abs(val_jac).sum(dim=(1, 2)))
            run_sp += torch.mean(mcp(val_jac))
            run_l0norm += torch.isclose(val_jac, torch.zeros((args.img_dim, args.latent_dim), device=device)).float().mean()

            b_it += 1

    return (run_recon_loss / b_it), (run_vol / b_it), (run_maxnorm / b_it), (run_l1norm / b_it), (run_sp / b_it), (run_l0norm / b_it)

def eval_synth(args, model, data_loader):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        run_recon_loss, run_vol, run_maxnorm, run_l1norm, run_sp, run_l0norm = 0, 0, 0, 0, 0, 0
        b_it = 0
        for x, z in data_loader:
            x = x.to(device)
            sh, xh = model(x)

            # Get reconstruction loss
            run_recon_loss += ((x - xh).square().mean()).item()

            # Get regularizers
            jacobian = torch.vmap(torch.func.jacfwd(model.decoder))(sh.flatten(1))
            run_vol += torch.mean(torch.det(jacobian.transpose(-1, -2) @ jacobian)).cpu().item()
            run_maxnorm += torch.mean(torch.amax(torch.abs(jacobian), dim=(1, 2))).cpu().item()
            run_l1norm += torch.mean(torch.abs(jacobian).sum(dim=(1, 2))).cpu().item()
            run_sp += torch.mean(mcp(jacobian)).cpu().item()
            run_l0norm += torch.isclose(jacobian, torch.zeros((args.m, args.latent_dim), device=device)).sum() / args.nobs

            # Save latents
            if b_it == 0:
                Z = z
                Zh = sh
            else:
                Z = torch.cat((Z, z))
                Zh = torch.cat((Zh, sh))

            b_it += 1

    # Get matrix of R2 scores
    r2_mat = r2(Z, Zh)
    r2_mat = torch.nn.functional.relu(torch.from_numpy(r2_mat))

    # Resolve permutation
    _, inds = utils.hungarian_algorithm(
        r2_mat.view(1, args.latent_dim, args.latent_dim) * -1
    )
    perm_r2_mat = r2_mat[:, inds[0][1]]

    # Get mean R2 on-diagonal
    r2_on = perm_r2_mat.diag()
    R2 = r2_on.mean().item()

    # Get matrix of MCC scores
    mcc_mat = mcc(Z, Zh)
    
    # Resolve permutation
    _, inds = utils.hungarian_algorithm(
        mcc_mat.view(1, args.latent_dim, args.latent_dim) * -1
    )
    perm_mcc_mat = mcc_mat[:, inds[0][1]]

    # Get mean MCC on-diagonal
    mcc_on = perm_mcc_mat.diag()
    MCC = mcc_on.mean().item()

    return (run_recon_loss / b_it), (run_vol / b_it), (run_maxnorm / b_it), (run_l1norm / b_it), (run_sp / b_it), (run_l0norm / b_it), R2, MCC


def plot_synth(args, model, data_loader, device=None, plot_x=True, filename=None):
    """
    Visualize the nonlinear ICA identifiability result by scatter plotting the true latents (z)
    and the estimated latents (sh) with color coding based on the angle of the data points.
    
    Args:
        args: Arguments containing model parameters
        model: Trained model
        data_loader: Data loader containing the synthetic data
        device: Device to run the model on
        plot_x: Whether to plot the input data
        filename: Path to save the plot data for later reproduction
    
    Returns:
        fig: Matplotlib figure containing the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Only proceed if latent dimension is 2
    if args.latent_dim != 2:
        print("Visualization only available for 2D latent space")
        return None
    
    model.eval()
    
    # Collect data
    Z_list = []
    Zh_list = []
    X_list = []
    
    with torch.no_grad():
        for x, z in data_loader:
            x = x.to(device)
            z = z.to(device)
            
            # Get model predictions
            sh, _ = model(x)
            sh = sh.reshape(sh.shape[0], args.latent_dim)
            
            # Save latents and inputs
            Z_list.append(z.cpu())
            Zh_list.append(sh.cpu())
            X_list.append(x.cpu())
    
    # Concatenate all batches
    Z = torch.cat(Z_list, dim=0).numpy()
    Zh = torch.cat(Zh_list, dim=0).numpy()
    X = torch.cat(X_list, dim=0).numpy()
    
    # Calculate angles for color coding
    angles_Z = np.arctan2(Z[:, 1], Z[:, 0])
    
    # Save data for plot reproduction if filename is provided
    if filename is not None:
        plot_data = {
            'Z': Z,
            'Zh': Zh,
            'X': X,
            'angles_Z': angles_Z,
            'plot_x': plot_x,
            'latent_dim': args.latent_dim
        }
        with open(filename, 'wb') as f:
            pickle.dump(plot_data, f)
    
    # Create figure
    if plot_x:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot input data as a 2D scatter plot
        scatter0 = axes[0].scatter(X[:, 0], X[:, 1], c=angles_Z, cmap='hsv', alpha=0.7)
        axes[0].set_title('Observations (X)')
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot true latents
        scatter1 = axes[1].scatter(Z[:, 0], Z[:, 1], c=angles_Z, cmap='hsv', alpha=0.7)
        axes[1].set_title('True Latent Space (Z)')
        axes[1].set_xlabel('z1')
        axes[1].set_ylabel('z2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot estimated latents with the same color coding
        scatter2 = axes[2].scatter(Zh[:, 0], Zh[:, 1], c=angles_Z, cmap='hsv', alpha=0.7)
        axes[2].set_title('Estimated Latent Space (Zh)')
        axes[2].set_xlabel('zh1')
        axes[2].set_ylabel('zh2')
        axes[2].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot true latents
        scatter1 = axes[0].scatter(Z[:, 0], Z[:, 1], c=angles_Z, cmap='hsv', alpha=0.7)
        axes[0].set_title('True Latent Space (Z)')
        axes[0].set_xlabel('z1')
        axes[0].set_ylabel('z2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot estimated latents with the same color coding
        scatter2 = axes[1].scatter(Zh[:, 0], Zh[:, 1], c=angles_Z, cmap='hsv', alpha=0.7)
        axes[1].set_title('Estimated Latent Space (Zh)')
        axes[1].set_xlabel('zh1')
        axes[1].set_ylabel('zh2')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
