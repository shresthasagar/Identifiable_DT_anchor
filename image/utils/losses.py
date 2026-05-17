import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, type=None, device='cpu'):
        super(GANLoss, self).__init__()
        if type is None:
            self.type = 'lsgan'
        else:
            self.type = type
        losses = {'lsgan': self.LSGAN_loss, 
                  'nsgan': self.NSGAN_loss,
                  'label_smooth_gan': self.LabelSmoothGAN_loss}
        
        self.loss = losses[self.type]
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, fake, real=None, is_disc=False, weight=None):
        return self.loss(fake, real, is_disc, weight=weight)
    
    def LSGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            w = weight if weight is not None else torch.ones_like(fake).to(self.device)
            return torch.mean((real-1)**2) + torch.mean(w * (fake**2))
        else:
            return torch.mean((fake-1)**2)
    
    def NSGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            bce_logit_loss_for_fake = nn.BCEWithLogitsLoss(weight=weight)
            return bce_logit_loss_for_fake(fake, torch.zeros_like(fake).to(self.device)) + self.bce_logit_loss(real, torch.ones_like(real).to(self.device))
        else:
            return self.bce_logit_loss(fake, torch.ones_like(fake).to(self.device))
    
    def LabelSmoothGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            bce_logit_loss_for_fake = nn.BCEWithLogitsLoss(weight=weight)
            return bce_logit_loss_for_fake(fake, torch.zeros_like(fake).to(self.device)) + self.bce_logit_loss(real, torch.ones_like(real).to(self.device)*0.9)
        else:
            return self.bce_logit_loss(fake, torch.ones_like(fake).to(self.device)*0.9)

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

class ReconsLoss(nn.Module):
    def __init__(self, type=None, device=None):
        super(ReconsLoss, self).__init__()
        if type is None:
            self.type = 'l1'
        else:
            self.type = type
        losses = {'l1': self.l1_loss, 
                  'l2': self.l2_loss}
        
        self.loss = losses[self.type]
        self.device = device

    def forward(self, fake, real):
        return self.loss(fake, real)
    
    def l1_loss(self, fake, real):
        return torch.mean(torch.abs(fake-real))
    
    def l2_loss(self, fake, real):
        return torch.mean((fake-real)**2)


def jacobian_l1_reg(model, x, num_samples=16):
    """
    Approximates the L1 norm of the Jacobian of `model` w.r.t. inputs `x`
    using stochastic output sampling.
    
    This version samples DIFFERENT random output indices for each sample in the batch,
    providing better variance reduction.
    """
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')

    x = x.requires_grad_(True)
    y = model(x)
    batch = x.shape[0]
    y_flat = y.view(batch, -1)
    total_outputs = y_flat.shape[1]
    if total_outputs == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    sample_count = min(num_samples, total_outputs)
    
    # Sample different random indices for each batch element
    if num_samples < total_outputs:
        idx = torch.stack([torch.randperm(total_outputs, device=x.device)[:sample_count] for _ in range(batch)])
    else:
        idx = torch.arange(0, total_outputs, device=x.device).unsqueeze(0).expand(batch, -1)
    
    reg = 0.0
    batch_indices = torch.arange(batch, device=x.device)

    for s in range(sample_count):
        batch_output_idx = idx[:, s]
        scalar = y_flat[batch_indices, batch_output_idx].sum()
        grad_x = torch.autograd.grad(
            scalar, x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        reg = reg + grad_x.abs().sum()

    reg = (total_outputs / sample_count) * reg
    reg = reg / batch
    return reg


def jacobian_fd_l1_reg(model, x, num_samples=16, probe_sparsity=0.1, sigma=1e-3):
    """
    Approximates the L1 norm of the Jacobian using sparse finite differences.
    
    This method:
    1. Samples a sparse binary mask z ~ Bernoulli(probe_sparsity) for each input dimension
    2. Samples a scalar perturbation magnitude e ~ N(0, sigma) per batch element
    3. Computes || (f(x + e*z) - f(x)) / e ||_1
    
    Args:
        model: The neural network
        x: Input tensor [batch, ...]
        num_samples: Number of random samples to average over
        probe_sparsity: Probability for Bernoulli mask (higher = more dimensions perturbed)
        sigma: Standard deviation for perturbation magnitude e
    
    Returns:
        Estimated Jacobian L1 norm proxy
    """
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')
    
    batch = x.shape[0]
    
    # Get baseline output
    y_base = model(x)
    y_base_flat = y_base.view(batch, -1)
    
    reg = 0.0
    epsilon = 1e-8

    for _ in range(num_samples):
        input_d = x.numel() // batch
        num_ones = int(probe_sparsity * input_d)
        z = torch.zeros_like(x)
        for b in range(batch):
            perm = torch.randperm(input_d, device=x.device)[:num_ones]
            z.view(batch, -1)[b, perm] = 1.0
        z = z.detach()
        
        e = (torch.randn_like(x, device=x.device, dtype=x.dtype) * sigma).detach()
        e = torch.where(e.abs() < epsilon, epsilon * torch.ones_like(e), e)
        
        x_perturbed = x.detach() + e * z
        y_perturbed = model(x_perturbed)
        y_perturbed_flat = y_perturbed.view(batch, -1)
        
        diff = (y_perturbed_flat - y_base_flat) / e.view(batch, -1)
        diff = diff / (y_base_flat.shape[1] * probe_sparsity)

        reg = reg + diff.abs().sum(dim=1).mean()
    
    return reg / num_samples


def jacobian_l1_exact(model, x):
    """
    Computes the EXACT L1 norm of the Jacobian using D JVPs (forward-mode AD).
    
    NOTE: This is expensive for high-dimensional inputs. Use only for logging/debugging,
    not for training loss computation.
    
    Args:
        model: The neural network
        x: Input tensor [batch, ...] with requires_grad=True
    
    Returns:
        Exact ||J||_1 averaged over batch
    """
    x = x.requires_grad_(True)
    batch = x.shape[0]
    x_flat = x.view(batch, -1)
    input_dim = x_flat.shape[1]
    
    if input_dim == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    total_l1 = 0.0
    
    for i in range(input_dim):
        e_i = torch.zeros_like(x_flat)
        e_i[:, i] = 1.0
        e_i = e_i.view_as(x)
        
        with torch.enable_grad():
            _, jvp_result = torch.autograd.functional.jvp(
                lambda inp: model(inp),
                (x,),
                (e_i,),
                create_graph=False
            )
        
        total_l1 = total_l1 + jvp_result.abs().sum() / batch
    
    return total_l1


def jacobian_reg(model, x, num_samples=16, norm_type='l1', p=1.0, probe_sparsity=0.5, sigma=1e-3):
    """
    Wrapper function for Jacobian regularization.
    
    Args:
        model: The neural network
        x: Input tensor with requires_grad=True
        num_samples: Number of samples for stochastic estimation
        norm_type: 
            - 'l1': L1 norm via stochastic output sampling (VJP-based)
            - 'fd_l1': L1 proxy via sparse finite differences (uses probe_sparsity, sigma)
        p: Unused (kept for backward compatibility)
        probe_sparsity: Probability for Bernoulli mask (only used when norm_type='fd_l1')
        sigma: Perturbation std dev (only used when norm_type='fd_l1')
    
    Returns:
        Jacobian regularization loss
    """
    if norm_type == 'l1':
        return jacobian_l1_reg(model, x, num_samples)
    elif norm_type == 'fd_l1':
        return jacobian_fd_l1_reg(model, x, num_samples, probe_sparsity, sigma)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Use 'l1' or 'fd_l1'.")
