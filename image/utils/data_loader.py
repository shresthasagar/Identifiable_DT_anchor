import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class LabeledDataset(Dataset):
    def __init__(self, 
                 input_folder, 
                 domain='A', 
                 keys=None,
                 train=True, 
                 rotate=0, 
                 new_size=28,
                 horizontal_flip=False,
                 use_paired_folder=False,
                 in_channels=None):
        """
        Args:
            input_folder: Path to dataset folder
            domain: 'A' or 'B'
            keys: List of label keys for conditional generation
            train: Whether to load train or test split
            rotate: Rotation angle in degrees
            new_size: Target image size
            horizontal_flip: Whether to apply random horizontal flip
            use_paired_folder: If True, use paired_trainA/paired_trainB folders
            in_channels: Number of channels (1 for grayscale, 3 for RGB, None for auto)
        """
        assert os.path.exists(input_folder), f'input_folder {input_folder} does not exist'
        self.in_channels = in_channels
        self.zero_pad = 'MNIST' in input_folder and new_size == 32

        self.keys = keys if keys is not None else ['dummy']

        # Determine folder path
        if domain == 'A':
            if use_paired_folder and train:
                folder_path = os.path.join(input_folder, 'paired_trainA')
                attribute_path = os.path.join(input_folder, 'paired_trainA_attr.csv')
                if not os.path.exists(attribute_path):
                    attribute_path = os.path.join(input_folder, 'trainA_attr.csv')
            else:
                folder_path = os.path.join(input_folder, 'trainA' if train else 'testA')
                attribute_path = os.path.join(input_folder, 'trainA_attr.csv' if train else 'testA_attr.csv')
        else:
            if use_paired_folder and train:
                folder_path = os.path.join(input_folder, 'paired_trainB')
                attribute_path = os.path.join(input_folder, 'paired_trainB_attr.csv')
                if not os.path.exists(attribute_path):
                    attribute_path = os.path.join(input_folder, 'trainB_attr.csv')
            else:
                folder_path = os.path.join(input_folder, 'trainB' if train else 'testB')
                attribute_path = os.path.join(input_folder, 'trainB_attr.csv' if train else 'testB_attr.csv')
        
        # Build transforms
        if self.zero_pad:
            self.base_transform = transforms.Compose([
                transforms.Pad([2], fill=0, padding_mode='constant'),
                transforms.ToTensor()
            ])
        else:
            base_list = [transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.BICUBIC)]
            rand_rotate = [transforms.RandomRotation((rotate, rotate), fill=255)] if rotate != 0 else []
            horizontal_flip_t = [transforms.RandomHorizontalFlip(p=0.5)] if (train and horizontal_flip) else []
            base_list = horizontal_flip_t + rand_rotate + base_list + [transforms.ToTensor()]
            self.base_transform = transforms.Compose(base_list)
        
        # Normalization for different channel counts
        self.normalize_1ch = transforms.Normalize((0.5,), (0.5,))
        self.normalize_3ch = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image_data = self._read_attr_file(attribute_path, folder_path)
        self.files = image_data['image_id'].values
        self.labels = image_data[self.keys].copy()
        for key in self.keys:
            self.labels[key] = list(map(lambda x: max(int(x), 0), self.labels[key]))
                
    def get_conditional_sizes(self):
        image_by_labels = []
        for key in self.keys:
            image_by_labels.append([self.files[j] for j in range(len(self.files)) if self.labels[key].iloc[j] == 1])
        return [len(imgs) for imgs in image_by_labels]

    def _read_attr_file(self, attr_path, image_dir):
        if os.path.exists(attr_path):
            f = open(attr_path)
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            columns = lines[0].split(',')
            lines = lines[1:]
            items = [line.split(',') for line in lines]
        else:
            print("Attribute file not found. Creating dummy attributes.")
            columns = ['image_id', 'dummy']
            items = [[x, 1] for x in os.listdir(image_dir)]

        df = pd.DataFrame(items, columns=columns)
        df['image_id'] = df['image_id'].map(lambda x: os.path.join(image_dir, x))
        return df

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = plt.imread(self.files[idx]).copy()
        
        # Handle grayscale vs RGB
        is_grayscale = len(image.shape) == 2
        
        if is_grayscale and self.in_channels == 3:
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        elif not is_grayscale and self.in_channels == 1:
            image = np.mean(image, axis=2).astype(image.dtype)

        if image.dtype == 'float32' and image.max() <= 1.0 and image.min() >= 0.0:
            image = (image * 255).astype(np.uint8)
        
        image = self.base_transform(Image.fromarray(image))
        
        # Apply normalization based on channels
        if image.shape[0] == 1:
            image = self.normalize_1ch(image)
        else:
            image = self.normalize_3ch(image)
        
        label = torch.tensor(self.labels.iloc[idx])
        return image, label


class PairedDataset(Dataset):
    """Dataset wrapper that pairs samples from two datasets index-wise."""
    def __init__(self, dataset_a, dataset_b, max_pairs=None, start_index=0):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.start_index = start_index
        max_len = min(len(self.dataset_a), len(self.dataset_b))
        if self.start_index >= max_len:
            raise ValueError(f'start_index {self.start_index} exceeds dataset length {max_len}')
        self.length = max_len - self.start_index
        if max_pairs is not None:
            self.length = min(self.length, max_pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        true_index = index + self.start_index
        img_a, label_a = self.dataset_a[true_index]
        img_b, label_b = self.dataset_b[true_index]
        return img_a, label_a, img_b, label_b


def _build_dataset(config, domain, train=True, rotate_angle=None, use_paired_folder=False):
    """Build dataset for MNIST domains."""
    if rotate_angle is None:
        rotate_angle = config.get('rotate_angle', 0)
    
    in_channels = config.get('in_channels', None)
    keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    if domain == 'mnist':
        dataset = LabeledDataset(
            input_folder=config['data_path'], 
            keys=keys, 
            domain='A', 
            rotate=rotate_angle, 
            new_size=config['new_size'], 
            horizontal_flip=False, 
            train=train,
            use_paired_folder=use_paired_folder,
            in_channels=in_channels
        )
    elif domain == 'rotatedmnist':
        dataset = LabeledDataset(
            input_folder=config['data_path'], 
            keys=keys, 
            domain='B', 
            rotate=rotate_angle, 
            new_size=config['new_size'], 
            horizontal_flip=False, 
            train=train,
            use_paired_folder=use_paired_folder,
            in_channels=in_channels
        )
    else:
        raise NotImplementedError(f'Domain {domain} not implemented. Use "mnist" or "rotatedmnist".')

    return dataset


def get_loader(config, domain, train=True, rotate_angle=None, paired=False):
    """Build and return DataLoader for specified domain.
    
    Args:
        config: Configuration dictionary
        domain: 'mnist' or 'rotatedmnist'
        train: Whether to load train or test split
        rotate_angle: Rotation angle in degrees
        paired: If True, disable shuffling to maintain sample correspondence
    """
    dataset = _build_dataset(config, domain, train=train, rotate_angle=rotate_angle)
    shuffle = train and not paired
    return DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers']
    )


def get_paired_loader(config, domain_a, domain_b, train=False, max_pairs=None, 
                      skip_first_n=0, rotate_angle_a=None, rotate_angle_b=None, paired=True):
    """Return dataloader that yields paired samples from both domains.
    
    Args:
        config: Configuration dictionary
        domain_a: First domain ('mnist')
        domain_b: Second domain ('rotatedmnist')
        train: Whether to load train or test split
        max_pairs: Maximum number of pairs to use
        skip_first_n: Skip first N pairs
        rotate_angle_a: Rotation angle for domain A
        rotate_angle_b: Rotation angle for domain B
        paired: If True, disable shuffling (default: True)
    """
    # Check for paired folders
    use_paired_folder = False
    if train:
        paired_a_path = os.path.join(config['data_path'], 'paired_trainA')
        paired_b_path = os.path.join(config['data_path'], 'paired_trainB')
        if os.path.exists(paired_a_path) and os.path.exists(paired_b_path):
            use_paired_folder = True
            print(f"[get_paired_loader] Using paired folders")
    
    dataset_a = _build_dataset(config, domain_a, train=train, rotate_angle=rotate_angle_a, 
                               use_paired_folder=use_paired_folder)
    dataset_b = _build_dataset(config, domain_b, train=train, rotate_angle=rotate_angle_b, 
                               use_paired_folder=use_paired_folder)
    paired_dataset = PairedDataset(dataset_a, dataset_b, max_pairs=max_pairs, start_index=skip_first_n)
    
    batch_size = config.get('paired_batch_size', max(1, config['batch_size'] // 4))
    shuffle = train and not paired
    return DataLoader(
        dataset=paired_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['num_workers']
    )
