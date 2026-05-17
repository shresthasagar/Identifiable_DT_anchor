"""
MNIST Dataset Preparation Script

Downloads MNIST dataset and creates the folder structure required for training:
- trainA/: Original MNIST images (domain 1)
- trainB/: Rotated MNIST images (domain 2)  
- testA/: Original MNIST test images
- testB/: Rotated MNIST test images

Usage:
    python utils/prepare_mnist_dataset.py --output_dir ./data/rotatedmnist --rotation -90
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from torchvision import datasets, transforms


def prepare_mnist_split(data, labels, output_folder, label_path, rotation=0):
    """
    Prepare a split of MNIST data (train or test).
    
    Args:
        data: MNIST image data (numpy array)
        labels: MNIST labels (numpy array)
        output_folder: Path to save images
        label_path: Path to save label CSV
        rotation: Rotation angle in degrees (multiples of 90)
    """
    data = data.numpy() if torch.is_tensor(data) else data
    labels = labels.numpy() if torch.is_tensor(labels) else labels
    
    # Rotate if specified
    if rotation != 0:
        k = int(rotation / 90)
        data = np.rot90(data, k=k, axes=(1, 2))
    
    # Convert to RGB format
    data = np.expand_dims(data, axis=1)
    data = np.repeat(data, 3, axis=1)
    data = data.astype(np.uint8)
    data = np.transpose(data, (0, 2, 3, 1))
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Save images
    print(f"Saving images to {output_folder}...")
    for i in trange(data.shape[0]):
        plt.imsave(os.path.join(output_folder, f'{i}.jpg'), data[i])
    
    # Create one-hot labels for digits 0-9
    one_hot_labels = np.ones((labels.shape[0], 10)) * -1
    one_hot_labels = one_hot_labels.astype(int)
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    
    # Save labels CSV
    print(f"Saving labels to {label_path}...")
    with open(label_path, 'w') as f:
        f.write('image_id,0,1,2,3,4,5,6,7,8,9\n')
        for i in range(one_hot_labels.shape[0]):
            f.write(f'{i}.jpg,' + ','.join(one_hot_labels[i].astype(str)) + '\n')
    
    print(f"Saved {data.shape[0]} images to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Prepare MNIST dataset for image translation')
    parser.add_argument('--output_dir', type=str, default='./data/rotatedmnist',
                        help='Output directory for the dataset')
    parser.add_argument('--rotation', type=int, default=90,
                        help='Rotation angle for domain B (multiples of 90, e.g., -90, 90, 180)')
    parser.add_argument('--download_dir', type=str, default='./data/MNIST_raw',
                        help='Directory to download raw MNIST data')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MNIST Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Rotation for domain B: {args.rotation} degrees")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)
    
    # Download MNIST
    print("\nDownloading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=args.download_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.download_dir, train=False, download=True, transform=transform)
    
    train_data = train_dataset.data
    train_labels = train_dataset.targets
    test_data = test_dataset.data
    test_labels = test_dataset.targets
    
    print(f"\nTrain set: {len(train_data)} images")
    print(f"Test set: {len(test_data)} images")
    
    # # Prepare domain A (original)
    # print("\n[Domain A] Preparing original MNIST (no rotation)...")
    # prepare_mnist_split(
    #     train_data, train_labels,
    #     os.path.join(args.output_dir, 'trainA'),
    #     os.path.join(args.output_dir, 'trainA_attr.csv'),
    #     rotation=0
    # )
    # prepare_mnist_split(
    #     test_data, test_labels,
    #     os.path.join(args.output_dir, 'testA'),
    #     os.path.join(args.output_dir, 'testA_attr.csv'),
    #     rotation=0
    # )
    
    # Prepare domain B (rotated)
    print(f"\n[Domain B] Preparing rotated MNIST ({args.rotation} degrees)...")
    prepare_mnist_split(
        train_data, train_labels,
        os.path.join(args.output_dir, 'trainB'),
        os.path.join(args.output_dir, 'trainB_attr.csv'),
        rotation=args.rotation
    )
    prepare_mnist_split(
        test_data, test_labels,
        os.path.join(args.output_dir, 'testB'),
        os.path.join(args.output_dir, 'testB_attr.csv'),
        rotation=args.rotation
    )
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDataset structure:")
    print(f"  {args.output_dir}/")
    print(f"    trainA/     - Original MNIST training images")
    print(f"    trainB/     - Rotated MNIST training images")
    print(f"    testA/      - Original MNIST test images")
    print(f"    testB/      - Rotated MNIST test images")
    print(f"\nTo use with training, set in your config:")
    print(f"  data_path: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
