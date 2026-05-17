#!/bin/bash
# export WANDB_DIR=/nfs/stak/users/shressag/hpc-share/Projects/wandb
export WANDB_DIR=./wandb

mkdir -p $WANDB_DIR


python synth_exp.py --n_layers 2 \
                    --hidden_dim 32 \
                    --lam_vol 0.0 \
                    --lam_recon 1.0 \
                    --lam_maxnorm 0.0 \
                    --lam_l1norm 1e-1 \
                    --lam_sp 0.0 \
                    --lam_gan 1.0 \
                    --disc_n_layers 2 \
                    --disc_hidden_dim 64 \
                    --disc_lr 1e-3 \
                    --batch_size 1024 \
                    --lr 1e-3 \
                    --num_iters 7000 \
                    --eval_iter 10 \
                    --plot_iter 1000 \
                    --nobs 30000 \
                    --latent_dim 2 \
                    --m 2 \
                    --cuda 0 \
                    --exp sparse_pnl \
                    --sp_ratio 0.5 \
                    --low_alpha 0.0001 \
                    --high_alpha 0.0002 \
                    --lam_anchor 1.0 \
                    --n_anchors 1 \
                    --sdist uniform_dependent \
                    --run_name "nica_dep_gan_anchor"