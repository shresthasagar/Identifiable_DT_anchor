
from model import Translator as Stargan_Translator
from model import Discriminator as MultiTaskDiscriminator
from model import FullyConnectedTranslator, FullyConnectedDiscriminator
import torch.nn as nn

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def get_model(config):
    """
    Builds generator and discriminator networks.
    
    Config options:
        - network_type: 'fcn' or 'cnn' (default: 'cnn')
        - in_channels: int (default: 3 for images)
        - fcn_hidden_dim: int (default: 1024)
        - gen: dict with use_adain, w_hpf, num_downsample for CNN
        - norm_type: 'instance' or 'channel_only' (default: 'instance')
    """
    network_type = config.get('network_type', 'cnn').lower()
    norm_type = config.get('norm_type', 'instance')
    
    effective_size = config['new_size']
    effective_channels = config.get('in_channels', 3)
    
    if norm_type != 'instance':
        print(f"[Model] Using normalization type: {norm_type}")
    
    if network_type == 'fcn':
        # Fully Connected Network
        fcn_config = config.get('fcn', {})
        gen_hidden_dim = fcn_config.get('hidden_dim', config.get('fcn_hidden_dim', 1024))
        disc_hidden_dim = fcn_config.get('disc_hidden_dim', config.get('fcn_discriminator_hidden_dim', gen_hidden_dim))

        g12 = FullyConnectedTranslator(
            img_size=effective_size,
            in_channels=effective_channels,
            hidden_dim=gen_hidden_dim
        )
        g21 = FullyConnectedTranslator(
            img_size=effective_size,
            in_channels=effective_channels,
            hidden_dim=gen_hidden_dim
        )

        print(f"[FCN] Generator hidden_dim={gen_hidden_dim}")

        d1 = [FullyConnectedDiscriminator(
            img_size=effective_size,
            num_domains=config['num_conditionals'],
            in_channels=effective_channels,
            hidden_dim=disc_hidden_dim
        )]
        d2 = [FullyConnectedDiscriminator(
            img_size=effective_size,
            num_domains=config['num_conditionals'],
            in_channels=effective_channels,
            hidden_dim=disc_hidden_dim
        )]
        
    elif network_type == 'cnn':
        # CNN (StarGAN-style)
        gen_config = config.get('gen', {})
        num_downsample = gen_config.get('num_downsample', None)
        use_adain = gen_config.get('use_adain', False)
        w_hpf = gen_config.get('w_hpf', 0)

        g12 = Stargan_Translator(
            img_size=effective_size, 
            w_hpf=w_hpf, 
            num_downsample=num_downsample, 
            use_adain=use_adain,
            in_channels=effective_channels,
            norm_type=norm_type
        )
        g21 = Stargan_Translator(
            img_size=effective_size, 
            w_hpf=w_hpf, 
            num_downsample=num_downsample, 
            use_adain=use_adain,
            in_channels=effective_channels,
            norm_type=norm_type
        )

        d1 = [MultiTaskDiscriminator(
            img_size=effective_size, 
            num_domains=config['num_conditionals'],
            in_channels=effective_channels,
            norm_type=norm_type
        )]
        d2 = [MultiTaskDiscriminator(
            img_size=effective_size, 
            num_domains=config['num_conditionals'],
            in_channels=effective_channels,
            norm_type=norm_type
        )]
    else:
        raise ValueError(f"Unknown network_type: {network_type}. Supported: 'cnn', 'fcn'")

    return g12, g21, d1, d2, None, None
