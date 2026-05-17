
import os
import time
from src.trainer import Trainer
from utils.data_loader import get_loader, get_paired_loader
import torch
import argparse
import yaml
import sys
import wandb
import datetime
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist.yaml', help='Path to the config file.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--eval', action='store_true', help='Evaluation mode: skip training, compute metrics and exit.')
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint for evaluation (optional, uses latest if not specified).')
opts = parser.parse_args()

# Set random seed
torch.manual_seed(opts.seed)
np.random.seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.safe_load(stream)

print(config) 

# Setup paths
config['model_path'] = os.path.join(config['model_path'], config['run_name'])
config['sample_path'] = os.path.join(config['sample_path'], config['run_name'])

# Create directories if not exist                                                           
if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])
if not os.path.exists(config['sample_path']):
    os.makedirs(config['sample_path'])

print('Preparing dataset...')

# Get domain-specific rotation angles (defaults to 0 if not specified)
rotate_angle_domain1 = config.get('rotate_domain1_angle', 0)
rotate_angle_domain2 = config.get('rotate_domain2_angle', 0)

train_loader1 = get_loader(config, domain=config['domain1'], train=True, rotate_angle=rotate_angle_domain1)
train_loader2 = get_loader(config, domain=config['domain2'], train=True, rotate_angle=rotate_angle_domain2)
test_loader1 = get_loader(config, domain=config['domain1'], train=False, rotate_angle=rotate_angle_domain1)
test_loader2 = get_loader(config, domain=config['domain2'], train=False, rotate_angle=rotate_angle_domain2)

display_count = 0
if not opts.debug:
    display_count = min(16, len(test_loader1.dataset), len(test_loader2.dataset))
    test_display_images1 = torch.stack([test_loader1.dataset[i][0] for i in range(display_count)]).cuda()
    test_display_images2 = torch.stack([test_loader2.dataset[i][0] for i in range(display_count)]).cuda()

paired_loader = None
paired_loader_iter = None

# Create paired loader if paired_loss_w > 0
if config.get('paired_loss_w', 0.0) > 0.0:
    paired_loader = get_paired_loader(
        config,
        domain_a=config['domain1'],
        domain_b=config['domain2'],
        train=True,
        max_pairs=config.get('paired_max_pairs'),
        skip_first_n=display_count,
        rotate_angle_a=rotate_angle_domain1,
        rotate_angle_b=rotate_angle_domain2,
        paired=True
    )
    paired_loader_iter = iter(paired_loader)

# Save sample images for debugging
print('Saving sample images...')
debug_dir = os.path.join(config['sample_path'], 'debug_samples')
os.makedirs(debug_dir, exist_ok=True)

# Save min(4, len(paired_loader)) paired samples from paired_loader if available,
# otherwise fallback to test_loader1/test_loader2
num_debug_samples = 4
if paired_loader is not None:
    paired_iter = iter(paired_loader)
    for i in range(num_debug_samples):
        try:
            batch = next(paired_iter)
        except StopIteration:
            break
        img1 = batch[0][0]  # domain1 image
        img2 = batch[2][0]  # domain2 image
        # Denormalize from [-1, 1] to [0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        save_image(img1, os.path.join(debug_dir, f'domain1_sample_{i}.png'))
        save_image(img2, os.path.join(debug_dir, f'domain2_sample_{i}.png'))
        # Save side by side
        combined = torch.cat([img1, img2], dim=2)
        save_image(combined, os.path.join(debug_dir, f'paired_sample_{i}.png'))

print(f'Saved debug images to {debug_dir}')

trainer = Trainer(config)
if config['adjust_class_imbalance']:
    cond_size1 = train_loader1.dataset.get_conditional_sizes()
    cond_size2 = train_loader2.dataset.get_conditional_sizes()
    trainer.set_gan_loss_weight(cond_size1, cond_size2)

run_name = config['run_name'] if config['run_name'] != '' else datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if config['use_wandb'] and not opts.debug:
    wandb.init(
        project='domain_translation',
        group=config['group_name'] if 'group_name' in config else None,
        name=run_name + f'-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}',
        resume=opts.resume,
        config=config
    )
        
iterations = 1

if opts.resume or opts.eval:
    checkpoint_path = opts.checkpoint_path if opts.checkpoint_path else os.path.join(config['model_path'], 'checkpoint-current.pt')
    if os.path.isfile(checkpoint_path):            
        print('Loading checkpoint {}'.format(checkpoint_path))
        iterations = trainer.load_checkpoint(checkpoint_path)
    else:
        print('No checkpoint found at {}'.format(checkpoint_path))
        sys.exit()

# Evaluation mode: generate samples and exit
if opts.eval:
    print('\n' + '='*60)
    print('EVALUATION MODE')
    print('='*60)
    
    if config.get('one_sided', False):
        merged1 = trainer.save_image_eval_one_sided(test_display_images1, 0)
    else:
        merged1, merged2 = trainer.save_image_eval(test_display_images1, test_display_images2, 0)
    
    print('\nEvaluation completed!')
    sys.exit(0)
            
iter_time_avg = 0.0
total_iters = config['train_iters']
pbar = tqdm(
    initial=iterations - 1,
    total=total_iters,
    desc=f'Training',
    unit='iter',
    ncols=120,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
)

while iterations <= config['train_iters']:
    for (images_1, labels_1), (images_2, labels_2) in zip(train_loader1, train_loader2):
        if config['lr_decay']:
            trainer.update_learning_rate(iterations)
        images_1, images_2 = images_1.cuda().detach(), images_2.cuda().detach()
        labels_1, labels_2 = labels_1.cuda(), labels_2.cuda()

        paired_samples = None
        if paired_loader_iter is not None:
            try:
                paired_batch = next(paired_loader_iter)
            except StopIteration:
                paired_loader_iter = iter(paired_loader)
                paired_batch = next(paired_loader_iter)
            paired_x1, _, paired_x2, _ = paired_batch
            paired_samples = (paired_x1.cuda().detach(), paired_x2.cuda().detach())

        if config['num_conditionals'] == 1:
            labels_1 = torch.ones(labels_1.shape[0], 1).cuda()
            labels_2 = torch.ones(labels_2.shape[0], 1).cuda()
        
        # Use one-sided training if enabled in config
        if config.get('one_sided', False):
            trainer.step_one_sided(images_1, labels_1, images_2, labels_2, paired_samples=paired_samples, iteration=iterations)
        else:
            trainer.step(images_1, labels_1, images_2, labels_2, paired_samples=paired_samples, iteration=iterations)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'iter': iterations,
            'lr': f'{trainer.g_optimizer.param_groups[0]["lr"]:.2e}' if trainer.g_optimizer else 'N/A'
        })

        # Log to console
        if iterations % config['console_log_steps'] == 0:        
            trainer.log_err_console(iterations)
            if not opts.debug:
                trainer.log_err_wandb(iterations)
                
        # Test model and save sampled images
        if iterations % config['test_sample_steps'] == 0 or iterations == 1:
            pbar.set_description('Evaluating')
            if config.get('one_sided', False):
                merged1 = trainer.save_image_eval_one_sided(test_display_images1, iterations)
                wandb_dict = {'A2B': wandb.Image(merged1)} if config['use_wandb'] and not opts.debug else {}
            else:
                merged1, merged2 = trainer.save_image_eval(test_display_images1, test_display_images2, iterations)
                wandb_dict = {'A2B': wandb.Image(merged1), 'B2A': wandb.Image(merged2)} if config['use_wandb'] and not opts.debug else {}
            
            if config['use_wandb'] and not opts.debug and wandb_dict:
                wandb.log(wandb_dict, step=iterations)
            pbar.set_description('Training')
            
        # Save model checkpoints
        if iterations % config['save_checkpoint_steps'] == 0:
            pbar.set_description('Saving checkpoint')
            trainer.save_checkpoint(os.path.join(config['model_path'], 'checkpoint-%d.pt' % iterations), iterations)
            trainer.save_checkpoint(os.path.join(config['model_path'], 'checkpoint-current.pt'), iterations)
            print('Saved model checkpoint')
            pbar.set_description('Training')

        iterations += 1
        if iterations > config['train_iters']:
            break
    
    if iterations > config['train_iters']:
        break

pbar.close()
print('Training completed!')
