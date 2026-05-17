import torch
import torch.nn as nn
import os
import numpy as np
from torch import optim
from src.trainer_utils import get_model, he_init
from torch.optim import lr_scheduler
import wandb
from utils.losses import GANLoss, ReconsLoss, jacobian_reg, jacobian_l1_exact, r1_reg
import copy
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class Trainer(object):
    def __init__(self, config):
        self.g12 = None
        self.g21 = None
        self.g12_ema = None
        self.g21_ema = None
        self.d1 = None
        self.d2 = None
        self.d1_all = None
        self.d2_all = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.config = config

        self.gan_loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

        self.gan_loss = GANLoss(config['gan_criterion'], device='cuda')
        self.recons_criterion = ReconsLoss(config['recons_criterion'])
        self.build_model()

        self.jacobian_loss_w = self.config.get('jacobian_loss_w', 0.0)
        # Separate weight for 2->1 direction (defaults to jacobian_loss_w for backward compatibility)
        self.jacobian_loss_w_2 = self.config.get('jacobian_loss_w_2', self.config.get('jacobian_loss_w', 0.0))
        self.jacobian_num_samples = self.config.get('jacobian_num_samples', 16)
        self.jacobian_norm_type = self.config.get('jacobian_norm_type', 'l1')  # 'l1', 'lp_jvp', 'l2', or 'fd_l1'
        self.jacobian_p = self.config.get('jacobian_p', 1.0)  # p value for LP norm
        # Parameters for fd_l1 (sparse finite difference) Jacobian regularization
        self.jacobian_probe_sparsity = self.config.get('jacobian_probe_sparsity', 0.5)
        self.jacobian_sigma = self.config.get('jacobian_sigma', 1e-3)  # Std dev for perturbation magnitude
        self.jacobian_start_iter = self.config.get('jacobian_start_iter', 0)  # Iteration to start jacobian regularization
        
        # Multiple generator updates per discriminator update
        n_gen_steps = self.config.get('n_gen_steps', 1)
        if n_gen_steps > 1:
            print(f"[Trainer] Using {n_gen_steps} generator updates per discriminator update")

        self.losses = {'loss_gen_total':         deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_adv_1':        deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_adv_2':        deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_cycrecon_x_1': deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_cycrecon_x_2': deque(maxlen=self.config['console_log_steps']),
                        'loss_paired_l1':        deque(maxlen=self.config['console_log_steps']),
                        'loss_paired_l1_reverse': deque(maxlen=self.config['console_log_steps']),
                        'loss_jacobian_12':      deque(maxlen=self.config['console_log_steps']),
                        'loss_jacobian_21':      deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_total':        deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_1':            deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_2':            deque(maxlen=self.config['console_log_steps']),
                        'r1_reg_1':              deque(maxlen=self.config['console_log_steps']), 
                        'r1_reg_2':              deque(maxlen=self.config['console_log_steps'])}
        
        # Exact Jacobian L1 computed at log time (expensive, so not every iteration)
        self.exact_jacobian_l1_12 = None
        self.exact_jacobian_l1_21 = None

    def set_gan_loss_weight(self, num_samples1, num_samples2):
        num_samples1 = torch.tensor(num_samples1).float()
        num_samples2 = torch.tensor(num_samples2).float()
        self.gan_loss_weights = (num_samples1 * num_samples2.sum()) / (num_samples2 * num_samples1.sum())

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12, self.g21, self.d1, self.d2, self.d1_all, self.d2_all = get_model(self.config)

        if 'he_init' in self.config.keys() and self.config['he_init']:
            self.g12.apply(he_init)
            self.g21.apply(he_init)
            for d in self.d1:
                d.apply(he_init)
            for d in self.d2:
                d.apply(he_init)
            if self.d1_all is not None:
                self.d1_all.apply(he_init)
            if self.d2_all is not None:
                self.d2_all.apply(he_init)

        self.g12_ema = copy.deepcopy(self.g12)    
        self.g21_ema = copy.deepcopy(self.g21)
        for (param1, param2) in zip(self.g12_ema.parameters(), self.g21_ema.parameters()):
            param1.requires_grad = False   
            param2.requires_grad = False
        self.g12_ema.eval()
        self.g21_ema.eval()

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = []
        for i in range(len(self.d1)):
            d_params += list(self.d1[i].parameters())
            d_params += list(self.d2[i].parameters())
        if self.config['dis_all_w'] > 0:
            d_params += list(self.d1_all.parameters())
            d_params += list(self.d2_all.parameters())
            
        self.g_optimizer = optim.Adam(g_params, self.config['lr'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])
        self.d_optimizer = optim.Adam(d_params, self.config['lr'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])
        
        def lambda_rule_exponential(num_calls):
            lr_l = 0.95**(max(0, num_calls +1 - (self.config['lr_start_step']/self.config['change_lr_every'])))
            return lr_l
        
        self.schedulers = [lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule_exponential),
                        lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda_rule_exponential)]

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.g12_ema.cuda()
            self.g21_ema.cuda()
            # self.g12 = torch.compile(self.g12)
            # self.g21 = torch.compile(self.g21)
            if self.config['dis_all_w'] > 0:
                self.d1_all.cuda()
                self.d2_all.cuda()
                # self.d1_all = torch.compile(self.d1_all)
                # self.d2_all = torch.compile(self.d2_all)
            for i in range(len(self.d1)):
                self.d1[i].cuda()
                self.d2[i].cuda()
                # self.d1[i] = torch.compile(self.d1[i])
                # self.d2[i] = torch.compile(self.d2[i])
    
    def merge_images_all(self, sources, targets, recons, k=10):
        labels = ['Source', 'Translation', 'Recons.']
        _, _, h, w = sources.shape
        row = min(int(np.sqrt(self.config['batch_size'])), 8)
        merged = np.zeros([3, row*h, row*w*3])
        for idx, (s, t, r) in enumerate(zip(sources, targets, recons)):
            if idx >= row*row:
                break
            i = idx // row
            j = idx % row
            # print(s.shape, i, j, h, w)
            merged[:, i*h:(i+1)*h, (j*3)*h:(j*3+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*3+1)*h:(j*3+2)*h] = t
            merged[:, i*h:(i+1)*h, (j*3+2)*h:(j*3+3)*h] = r
        return merged.transpose(1,2,0)


    def merge_images_all_plt(self, sources, targets, recons):
        num_images, n_cols = sources.shape[0], 3
        sources, targets, recons = sources*0.5 + 0.5, targets*0.5 + 0.5, recons*0.5 + 0.5
        fig = plt.figure(figsize=(n_cols*10, (num_images)*3))
        grid = ImageGrid(fig, 111, nrows_ncols=(num_images, n_cols), axes_pad=0.1)
        fontsize = 15
        for i in range(num_images):
            grid[i*n_cols].imshow(np.transpose(sources[i],(1, 2, 0)))
            grid[i*n_cols+1].imshow(np.transpose(targets[i],(1, 2, 0)))
            grid[i*n_cols+2].imshow(np.transpose(recons[i],(1, 2, 0)))
            
            # set title for the first row
            if i == 0:
                grid[i*n_cols].set_title('Source', fontsize=fontsize)
                grid[i*n_cols+1].set_title('Translation', fontsize=fontsize)
                grid[i*n_cols+2].set_title('Recons.', fontsize=fontsize)

        # remove the x and y ticks
        for ax in grid:
            ax.set_xticks([])
            ax.set_yticks([])
            
        return fig


    def save_image_eval(self, test_domain1, test_domain2, iterations):
        """Evaluate and save sample translations."""
        fake_2 = self.g12_ema(test_domain1)
        fake_1 = self.g21_ema(test_domain2)
        recons_1 = self.g21_ema(fake_2)
        recons_2 = self.g12_ema(fake_1)
        
        domain1 = test_domain1.clamp_(-1, 1).detach().cpu().numpy()
        domain2 = test_domain2.clamp_(-1, 1).detach().cpu().numpy()
        fake_1 = fake_1.clamp_(-1, 1).detach().cpu().numpy()
        fake_2 = fake_2.clamp_(-1, 1).detach().cpu().numpy()
        recons_1 = recons_1.clamp_(-1, 1).detach().cpu().numpy()
        recons_2 = recons_2.clamp_(-1, 1).detach().cpu().numpy()
        
        fig1 = self.merge_images_all(domain1, fake_2, recons_1)
        fig2 = self.merge_images_all(domain2, fake_1, recons_2)

        return fig1, fig2

    def save_image_eval_one_sided(self, test_domain1, iterations):
        """One-sided evaluation: only g_12 (domain1 -> domain2)."""
        fake_2 = self.g12_ema(test_domain1)
        
        domain1 = test_domain1.clamp_(-1, 1).detach().cpu().numpy()
        fake_2 = fake_2.clamp_(-1, 1).detach().cpu().numpy()
        
        fig1 = self.merge_images_two(domain1, fake_2)

        return fig1
    
    def merge_images_two(self, domain1, fake_domain2):
        """Merge images for one-sided visualization (input, output)."""
        n = domain1.shape[0]
        fig = plt.figure(figsize=(n * 2, 4))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, n), axes_pad=0.05)
        
        for i in range(n):
            img1 = (domain1[i].transpose(1, 2, 0) + 1) / 2
            img2 = (fake_domain2[i].transpose(1, 2, 0) + 1) / 2
            
            if img1.shape[-1] == 1:
                img1 = img1.squeeze(-1)
                img2 = img2.squeeze(-1)
                grid[i].imshow(img1, cmap='gray')
                grid[i + n].imshow(img2, cmap='gray')
            else:
                grid[i].imshow(img1)
                grid[i + n].imshow(img2)
            
            grid[i].axis('off')
            grid[i + n].axis('off')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data

    def step(self, x_1, l_1, x_2, l_2, paired_samples=None, iteration=0):
        """Full training step for bidirectional translation."""
        # Update Discriminator
        self.d_optimizer.zero_grad()
        
        x_12 = self.g12(x_1)
        x_21 = self.g21(x_2)
        
        self.loss_dis_1, self.loss_dis_2 = 0.0, 0.0
        self.r1_reg_1, self.r1_reg_2 = torch.tensor([0.0]).cuda(), torch.tensor([0.0]).cuda()
        
        if self.config['r1_reg_w'] > 0:
            x_1.requires_grad_()
            x_2.requires_grad_()

        out1, _ = self.d1[0](x_1, l_1)
        out2, _ = self.d2[0](x_2, l_2)
        out_fake1, cond_id1 = self.d1[0](x_21.detach(), l_2)
        weight1 = 1 / torch.tensor([self.gan_loss_weights[i] for i in cond_id1]).to(l_1.device)
        out_fake2, cond_id2 = self.d2[0](x_12.detach(), l_1)
        weight2 = torch.tensor([self.gan_loss_weights[i] for i in cond_id2]).to(l_1.device)

        self.loss_dis_1 += self.gan_loss(out_fake1, real=out1, is_disc=True, weight=weight1)
        self.loss_dis_2 += self.gan_loss(out_fake2, real=out2, is_disc=True, weight=weight2)
        if self.config['r1_reg_w'] > 0:    
            self.r1_reg_1 += r1_reg(out1, x_1)
            self.r1_reg_2 += r1_reg(out2, x_2)

        self.loss_dis_total = self.config['dis_w'] * self.loss_dis_1 + \
                                self.config['dis_w'] * self.loss_dis_2 + \
                                self.config['r1_reg_w'] * self.r1_reg_1 + \
                                self.config['r1_reg_w'] * self.r1_reg_2
        
        self.loss_dis_total.backward()
        self.d_optimizer.step()
        
        # Update Generator
        self.g_optimizer.zero_grad()
        
        x_12 = self.g12(x_1)
        x_21 = self.g21(x_2)
        x_121 = self.g21(x_12)
        x_212 = self.g12(x_21)
        
        # Cycle reconstruction loss
        self.loss_gen_cycrecon_x_1 = self.recons_criterion(x_121, x_1) 
        self.loss_gen_cycrecon_x_2 = self.recons_criterion(x_212, x_2)
        
        # GAN loss
        self.loss_gen_adv_1, self.loss_gen_adv_2 = 0.0, 0.0
        out_fake1, _ = self.d1[0](x_21, l_2)
        out_fake2, _ = self.d2[0](x_12, l_1)
        self.loss_gen_adv_1 += self.gan_loss(out_fake1, is_disc=False)
        self.loss_gen_adv_2 += self.gan_loss(out_fake2, is_disc=False)      

        # Paired Loss
        self.loss_paired_l1 = torch.tensor(0.0, device=x_1.device)
        self.loss_paired_l1_reverse = torch.tensor(0.0, device=x_1.device)
        if paired_samples is not None and self.config.get('paired_loss_w', 0.0) > 0.0:
            paired_x1, paired_x2 = paired_samples
            self.loss_paired_l1 = self.recons_criterion(self.g12(paired_x1), paired_x2)
            self.loss_paired_l1_reverse = self.recons_criterion(self.g21(paired_x2), paired_x1)

        # Jacobian regularization
        self.loss_jacobian_12 = torch.tensor(0.0, device=x_1.device)
        self.loss_jacobian_21 = torch.tensor(0.0, device=x_1.device)
        if self.jacobian_loss_w > 0.0 and iteration >= self.jacobian_start_iter:
            x1_jac = x_1.detach().requires_grad_(True)
            self.loss_jacobian_12 = jacobian_reg(self.g12, x1_jac, self.jacobian_num_samples, self.jacobian_norm_type, self.jacobian_p, self.jacobian_probe_sparsity, self.jacobian_sigma)
        if self.jacobian_loss_w_2 > 0.0 and iteration >= self.jacobian_start_iter:
            x2_jac = x_2.detach().requires_grad_(True)
            self.loss_jacobian_21 = jacobian_reg(self.g21, x2_jac, self.jacobian_num_samples, self.jacobian_norm_type, self.jacobian_p, self.jacobian_probe_sparsity, self.jacobian_sigma)
            
            self.loss_gen_total = self.config['gen_w'] * self.loss_gen_adv_1 + \
                                self.config['gen_w'] * self.loss_gen_adv_2 + \
                                self.config['recons_w'] * self.loss_gen_cycrecon_x_1 + \
                                self.config['recons_w'] * self.loss_gen_cycrecon_x_2
            if self.config.get('paired_loss_w', 0.0) > 0.0 and paired_samples is not None:
                self.loss_gen_total = self.loss_gen_total + \
                                      self.config['paired_loss_w'] * (self.loss_paired_l1 + self.loss_paired_l1_reverse)
            # Add Jacobian regularization with separate weights for each direction
            if self.jacobian_loss_w > 0.0:
                self.loss_gen_total = self.loss_gen_total + self.jacobian_loss_w * self.loss_jacobian_12
            if self.jacobian_loss_w_2 > 0.0:
                self.loss_gen_total = self.loss_gen_total + self.jacobian_loss_w_2 * self.loss_jacobian_21
            self.loss_gen_total.backward()
            self.g_optimizer.step()

        if self.config['use_ema']:
            self.moving_average(self.g12, self.g12_ema, beta=0.999)
            self.moving_average(self.g21, self.g21_ema, beta=0.999)
        else:
            self.moving_average(self.g12, self.g12_ema, beta=0.0)
            self.moving_average(self.g21, self.g21_ema, beta=0.0)

        # Update average Losses (log only the last generator step's losses)
        self.losses['loss_gen_total'].append(self.loss_gen_total.item())
        self.losses['loss_gen_adv_1'].append(self.loss_gen_adv_1.item())
        self.losses['loss_gen_adv_2'].append(self.loss_gen_adv_2.item())
        self.losses['loss_gen_cycrecon_x_1'].append(self.loss_gen_cycrecon_x_1.item())
        self.losses['loss_gen_cycrecon_x_2'].append(self.loss_gen_cycrecon_x_2.item())
        self.losses['loss_dis_total'].append(self.loss_dis_total.item())
        self.losses['loss_dis_1'].append(self.loss_dis_1.item())
        self.losses['loss_dis_2'].append(self.loss_dis_2.item())
        self.losses['r1_reg_1'].append(self.r1_reg_1.item())
        self.losses['r1_reg_2'].append(self.r1_reg_2.item())
        self.losses['loss_paired_l1'].append(self.loss_paired_l1.item())
        self.losses['loss_paired_l1_reverse'].append(self.loss_paired_l1_reverse.item())
        self.losses['loss_jacobian_12'].append(self.loss_jacobian_12.item())
        self.losses['loss_jacobian_21'].append(self.loss_jacobian_21.item())

    def step_one_sided(self, x_1, l_1, x_2, l_2, paired_samples=None, iteration=0):
        """One-sided training: only trains g_12 and d2."""
        # Update Discriminator (only D2)
        self.d_optimizer.zero_grad()
        
        x_12 = self.g12(x_1)
        
        self.loss_dis_1, self.loss_dis_2 = 0.0, 0.0
        self.r1_reg_1, self.r1_reg_2 = torch.tensor([0.0]).cuda(), torch.tensor([0.0]).cuda()
        
        if self.config['r1_reg_w'] > 0:
            x_2.requires_grad_()

        out2, _ = self.d2[0](x_2, l_2)
        out_fake2, cond_id2 = self.d2[0](x_12.detach(), l_1)
        weight2 = torch.tensor([self.gan_loss_weights[i] for i in cond_id2]).to(l_1.device)
        self.loss_dis_2 += self.gan_loss(out_fake2, real=out2, is_disc=True, weight=weight2)
        if self.config['r1_reg_w'] > 0:    
            self.r1_reg_2 += r1_reg(out2, x_2)

        self.loss_dis_total = self.config['dis_w'] * self.loss_dis_2 + \
                                self.config['r1_reg_w'] * self.r1_reg_2
        
        self.loss_dis_total.backward()
        self.d_optimizer.step()
        
        # Update Generator (only g_12)
        self.g_optimizer.zero_grad()
        
        x_12 = self.g12(x_1)
        x_121 = self.g21(x_12)  # Use g_21 for invertibility regularization
        
        self.loss_gen_cycrecon_x_1 = self.recons_criterion(x_121, x_1)
        self.loss_gen_cycrecon_x_2 = torch.tensor(0.0, device=x_1.device)
        
        # GAN loss
        self.loss_gen_adv_1, self.loss_gen_adv_2 = 0.0, 0.0
        out_fake2, _ = self.d2[0](x_12, l_1)
        self.loss_gen_adv_2 += self.gan_loss(out_fake2, is_disc=False)

        # Paired loss
        self.loss_paired_l1 = torch.tensor(0.0, device=x_1.device)
        self.loss_paired_l1_reverse = torch.tensor(0.0, device=x_1.device)
        if paired_samples is not None and self.config.get('paired_loss_w', 0.0) > 0.0:
            paired_x1, paired_x2 = paired_samples
            self.loss_paired_l1 = self.recons_criterion(self.g12(paired_x1), paired_x2)

        # Jacobian loss
        self.loss_jacobian_12 = torch.tensor(0.0, device=x_1.device)
        self.loss_jacobian_21 = torch.tensor(0.0, device=x_1.device)
        if self.jacobian_loss_w > 0.0 and iteration >= self.jacobian_start_iter:
            x1_jac = x_1.detach().requires_grad_(True)
            self.loss_jacobian_12 = jacobian_reg(self.g12, x1_jac, self.jacobian_num_samples, self.jacobian_norm_type, self.jacobian_p, self.jacobian_probe_sparsity, self.jacobian_sigma)
        
        self.loss_gen_total = self.config['gen_w'] * self.loss_gen_adv_2 + \
                            self.config['recons_w'] * self.loss_gen_cycrecon_x_1
        
        if self.config.get('paired_loss_w', 0.0) > 0.0 and paired_samples is not None:
            self.loss_gen_total += self.config['paired_loss_w'] * self.loss_paired_l1
        
        if self.jacobian_loss_w > 0.0:
            self.loss_gen_total += self.jacobian_loss_w * self.loss_jacobian_12
        
        self.loss_gen_total.backward()
        self.g_optimizer.step()

        # EMA update
        beta = 0.999 if self.config['use_ema'] else 0.0
        self.moving_average(self.g12, self.g12_ema, beta)

        # Update average Losses (log only the last generator step's losses)
        self.losses['loss_gen_total'].append(self.loss_gen_total.item())
        self.losses['loss_gen_adv_1'].append(0.0)
        self.losses['loss_gen_adv_2'].append(self.loss_gen_adv_2.item())
        self.losses['loss_gen_cycrecon_x_1'].append(self.loss_gen_cycrecon_x_1.item())
        self.losses['loss_gen_cycrecon_x_2'].append(0.0)
        self.losses['loss_dis_total'].append(self.loss_dis_total.item())
        self.losses['loss_dis_1'].append(0.0)
        self.losses['loss_dis_2'].append(self.loss_dis_2.item())
        self.losses['r1_reg_1'].append(0.0)
        self.losses['r1_reg_2'].append(self.r1_reg_2.item())
        self.losses['loss_paired_l1'].append(self.loss_paired_l1.item())
        self.losses['loss_paired_l1_reverse'].append(0.0)
        self.losses['loss_jacobian_12'].append(self.loss_jacobian_12.item())
        self.losses['loss_jacobian_21'].append(0.0)
        
    
    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def sample(self, x_1, x_2):
        x_121, x_212, x_21, x_12 = [], [], [], []
        for i in range(x_1.size(0)):
            x_21.append(self.g21_ema(x_2[i].unsqueeze(0)))
            x_12.append(self.g12_ema(x_1[i].unsqueeze(0)))
            x_121.append(self.g21_ema(x_12[-1]))
            x_212.append(self.g12_ema(x_21[-1]))
        x_121, x_212 = torch.cat(x_121), torch.cat(x_212)
        x_12, x_21 = torch.cat(x_12), torch.cat(x_21)
        
        return x_1.detach().cpu(), x_121.detach().cpu(), x_12.detach().cpu(), x_2.detach().cpu(), x_212.detach().cpu(), x_21.detach().cpu()

    def update_learning_rate(self, iterations):
        if iterations%self.config['change_lr_every'] == 0: 
            if self.schedulers is not None:
                for scheduler in self.schedulers:
                    scheduler.step()
            
    def save_checkpoint(self, filename, iterations):
        params = {
            'g12': self.g12.state_dict(),
            'g21': self.g21.state_dict(),
            'g12_ema': self.g12_ema.state_dict(),
            'g21_ema': self.g21_ema.state_dict(),
            'd1': [net.state_dict() for net in self.d1],
            'd2': [net.state_dict() for net in self.d2],
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'step': iterations
        }
        torch.save(params, filename)
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.g12.load_state_dict(checkpoint['g12'])
            self.g21.load_state_dict(checkpoint['g21'])
            self.g12_ema.load_state_dict(checkpoint['g12_ema'])
            self.g21_ema.load_state_dict(checkpoint['g21_ema'])
            for i in range(len(self.d1)):
                self.d1[i].load_state_dict(checkpoint['d1'][i])
                self.d2[i].load_state_dict(checkpoint['d2'][i])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            return checkpoint['step']
        else:
            print(f'Loading Checkpoint Failed. Checkpoint {checkpoint_path} does not exist.')
                     
    @torch.no_grad()
    def compute_exact_jacobian_l1(self, x_samples, one_sided=False):
        """
        Compute the exact Jacobian L1 norm using D forward passes (JVPs).
        
        This is expensive so should only be called at log time, not every iteration.
        
        Args:
            x_samples: Input samples [batch, ...] to compute Jacobian on
            one_sided: If True, only compute g_12 (A->B), else compute both directions
        
        Returns:
            Dictionary with exact Jacobian L1 values
        """
        self.g12_ema.eval()
        self.g21_ema.eval()
        
        self.exact_jacobian_l1_12 = jacobian_l1_exact(self.g12_ema, x_samples).item()
        result = {'exact_jacobian_l1_12': self.exact_jacobian_l1_12}
        
        if not one_sided:
            self.exact_jacobian_l1_21 = jacobian_l1_exact(self.g21_ema, x_samples).item()
            result['exact_jacobian_l1_21'] = self.exact_jacobian_l1_21
        
        return result
    
    def log_err_wandb(self, iterations, iter_time=None):
        if self.config['use_wandb']:
            result = {}
            for key in self.losses:
                result[key] = np.mean(self.losses[key])
            if iter_time is not None:
                result['iter_time'] = iter_time
            # Add exact Jacobian L1 if computed
            if self.exact_jacobian_l1_12 is not None:
                result['exact_jacobian_l1_12'] = self.exact_jacobian_l1_12
            if self.exact_jacobian_l1_21 is not None:
                result['exact_jacobian_l1_21'] = self.exact_jacobian_l1_21
            wandb.log(result, step=iterations)
        
    def _get_loss_value(self, loss):
        """Helper to get loss value whether it's a tensor or float."""
        return loss.item() if hasattr(loss, 'item') else loss
    
    def log_err_console(self, it, iter_time=None):
        time_str = '' if iter_time is None else f', iter_time: {iter_time:.4f}s'
        exact_jac_str = ''
        if self.exact_jacobian_l1_12 is not None:
            exact_jac_str += f', exact_jac_l1_12: {self.exact_jacobian_l1_12:.4f}'
        if self.exact_jacobian_l1_21 is not None:
            exact_jac_str += f', exact_jac_l1_21: {self.exact_jacobian_l1_21:.4f}'
        print('iter: %d, loss_gen_total: %.4f, loss_gen_adv_1: %.4f, loss_gen_adv_2: %.4f, loss_gen_cycrecon_x_1: %.4f, loss_gen_cycrecon_x_2: %.4f, loss_paired_l1: %.4f, loss_paired_l1_reverse: %.4f, loss_jacobian_12: %.4f, loss_jacobian_21: %.4f, loss_dis_total: %.4f, loss_dis_1: %.4f, loss_dis_2: %.4f%s%s' %  \
                    (it, self._get_loss_value(self.loss_gen_total), self._get_loss_value(self.loss_gen_adv_1), self._get_loss_value(self.loss_gen_adv_2), self._get_loss_value(self.loss_gen_cycrecon_x_1), self._get_loss_value(self.loss_gen_cycrecon_x_2), self._get_loss_value(self.loss_paired_l1), self._get_loss_value(self.loss_paired_l1_reverse), self._get_loss_value(self.loss_jacobian_12), self._get_loss_value(self.loss_jacobian_21), self._get_loss_value(self.loss_dis_total), self._get_loss_value(self.loss_dis_1), self._get_loss_value(self.loss_dis_2), exact_jac_str, time_str))