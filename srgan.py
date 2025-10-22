import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg19
import rasterio
import numpy as np
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==================== Utility Functions ====================
def calculate_metrics(sr_images, hr_images):
    """
    Calculate PSNR and SSIM metrics between SR and HR images.
    
    Args:
        sr_images: Super-resolved images (batch, channels, H, W)
        hr_images: High-resolution ground truth images (batch, channels, H, W)
    
    Returns:
        avg_psnr: Average PSNR across batch
        avg_ssim: Average SSIM across batch
    """
    sr_np = sr_images.detach().cpu().numpy()
    hr_np = hr_images.detach().cpu().numpy()
    
    psnr_values = []
    ssim_values = []
    
    for i in range(sr_np.shape[0]):
        # Convert to 2D if single channel
        sr_img = sr_np[i, 0] if sr_np.shape[1] == 1 else sr_np[i]
        hr_img = hr_np[i, 0] if hr_np.shape[1] == 1 else hr_np[i]
        
        # Ensure values are in [0, 1] range
        sr_img = np.clip(sr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # Calculate PSNR
        psnr_val = psnr(hr_img, sr_img, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Calculate SSIM
        ssim_val = ssim(hr_img, sr_img, data_range=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values)


def save_comparison_images(lr_imgs, sr_imgs, hr_imgs, epoch, save_dir='results', num_samples=4):
    """
    Save comparison images showing LR, SR, and HR side by side.
    
    Args:
        lr_imgs: Low-resolution input images
        sr_imgs: Super-resolved images
        hr_imgs: High-resolution ground truth images
        epoch: Current epoch number
        save_dir: Directory to save images
        num_samples: Number of samples to save
    """
    os.makedirs(save_dir, exist_ok=True)
    sr_imgs = (sr_imgs > 0.2).float()  # Binarize SR images for clearer visualizations
    # Convert to numpy and take first num_samples
    lr_np = lr_imgs[:num_samples].detach().cpu().numpy()
    sr_np = sr_imgs[:num_samples].detach().cpu().numpy()
    hr_np = hr_imgs[:num_samples].detach().cpu().numpy()
    
    num_samples = min(num_samples, lr_np.shape[0])

    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    print(lr_np.shape, sr_np.shape, hr_np.shape)
    for i in range(num_samples):
        # LR image (take first channel if multi-channel)
        lr_img = lr_np[i, -1] if lr_np.shape[1] > 1 else lr_np[i, 0]
        sr_img = sr_np[i, 0]
        hr_img = hr_np[i, 0]
        
        # Clip values to [0, 1]
        lr_img = np.clip(lr_img, 0, 1)
        sr_img = np.clip(sr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # Calculate metrics for this sample
        psnr_val = psnr(hr_img, sr_img, data_range=1.0)
        ssim_val = ssim(hr_img, sr_img, data_range=1.0)
        
        # Plot LR
        axes[i, 0].imshow(lr_img, cmap='gray')
        axes[i, 0].set_title(f'LR ({lr_img.shape[0]}x{lr_img.shape[1]})')
        axes[i, 0].axis('off')
        
        # Plot SR
        axes[i, 1].imshow(sr_img, cmap='gray')
        axes[i, 1].set_title(f'SR (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})')
        axes[i, 1].axis('off')
        
        # Plot HR
        axes[i, 2].imshow(hr_img, cmap='gray')
        axes[i, 2].set_title(f'HR ({hr_img.shape[0]}x{hr_img.shape[1]})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison images to {save_dir}/comparison_epoch_{epoch:03d}.png")


def plot_training_curves(history, save_dir='results'):
    """
    Plot training curves for losses and metrics.
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator Loss
    axes[0, 0].plot(history['g_loss'], label='G Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Loss
    axes[0, 1].plot(history['d_loss'], label='D Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSNR
    if 'psnr' in history and history['psnr']:
        axes[1, 0].plot(history['psnr'], label='PSNR', color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # SSIM
    if 'ssim' in history and history['ssim']:
        axes[1, 1].plot(history['ssim'], label='SSIM', color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Structural Similarity Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_dir}/training_curves.png")


# ==================== Dataset ====================
class SEVIRIModisDataset(Dataset):
    def __init__(self, root_dir, transform_lr=None, transform_hr=None, 
                 seviri_channels=None, skip_black_targets=True, min_variance=1e-6):
        """
        Dataset for SEVIRI (LR) and MODIS (HR) pairs.
        Files are named like: PREFIX_seviri.tif and PREFIX_modis.tif
        
        Args:
            root_dir: Path to dataset directory
            transform_lr: Optional transforms for LR images
            transform_hr: Optional transforms for HR images
            seviri_channels: List of channel indices to use (e.g., [0,1,2,5,8]). 
                           If None, uses all channels.
            skip_black_targets: If True, skip pairs where MODIS is all zeros
            min_variance: Minimum variance threshold for valid MODIS images
        """
        self.root_dir = Path(root_dir)
        self.seviri_channels = seviri_channels
        self.skip_black_targets = skip_black_targets
        self.min_variance = min_variance
        
        print(f"Loading dataset from: {self.root_dir}")
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {self.root_dir}")
        
        # Find all SEVIRI files (these will be our LR images)
        all_seviri_files = sorted(list(self.root_dir.glob('*_seviri.tif')))
        
        if not all_seviri_files:
            raise ValueError(f"No SEVIRI files (*_seviri.tif) found in {self.root_dir}")
        
        # Create pairs and filter if needed
        self.pairs = []
        skipped_black = 0
        skipped_missing = 0
        
        for seviri_path in all_seviri_files:
            prefix = seviri_path.stem.replace('_seviri', '')
            modis_path = self.root_dir / f"{prefix}_modis.tif"
            
            if not modis_path.exists():
                skipped_missing += 1
                continue
            
            # Check if MODIS target is valid
            if self.skip_black_targets:
                try:
                    with rasterio.open(modis_path) as src:
                        modis_data = src.read()
                    
                    # Check if image is all zeros or has very low variance
                    if np.all(modis_data == 0):# or np.var(modis_data) < self.min_variance:
                        skipped_black += 1
                        continue
                except:
                    skipped_missing += 1
                    continue
            
            self.pairs.append((seviri_path, modis_path))
        
        if not self.pairs:
            print(f"\nERROR: No valid pairs found!")
            print(f"  Total SEVIRI files: {len(all_seviri_files)}")
            print(f"  Skipped (missing MODIS): {skipped_missing}")
            print(f"  Skipped (black/low variance): {skipped_black}")
            print(f"\nTry setting skip_black_targets=False or check your data directory.")
            raise ValueError(f"No valid SEVIRI-MODIS pairs found in {self.root_dir}")
        
        print(f"Dataset initialized with {len(self.pairs)} valid pairs")
        if skipped_black > 0:
            print(f"  Skipped {skipped_black} pairs with black/low-variance MODIS targets")
        if skipped_missing > 0:
            print(f"  Skipped {skipped_missing} pairs with missing MODIS files")
        if self.seviri_channels:
            print(f"  Using SEVIRI channels: {self.seviri_channels}")
        else:
            print(f"  Using all SEVIRI channels")
        print(f"Example pair: {self.pairs[0][0].name} <-> {self.pairs[0][1].name}")
        
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
    
    def __len__(self):
        return len(self.pairs)
        
    #def __len__(self):
    #    return len(self.seviri_files)
    
    def __getitem__(self, idx):
        seviri_path, modis_path = self.pairs[idx]
        
        try:
            # Load SEVIRI (LR) - using rasterio for GeoTIFF
            #import rasterio
            with rasterio.open(seviri_path) as src:
                seviri = src.read()  # Shape: (bands, H, W)
            
            # Select specific channels if specified
            if self.seviri_channels is not None:
                seviri = seviri[self.seviri_channels, :, :]
            
            # Load MODIS (HR)
            with rasterio.open(modis_path) as src:
                modis = src.read()  # Shape: (bands, H, W)
        except Exception as e:
            print(f"Error loading {seviri_path.name}: {e}")
            raise
        
        # Convert to torch tensors
        seviri = torch.from_numpy(seviri).float()
        modis = torch.from_numpy(modis).float()

                # CRITICAL: Process cloud mask channel (index 11 in full 12-channel SEVIRI)
        # Convert values of 2 to 1 for binary cloud mask
        if self.seviri_channels is None:
            # Using all channels, so channel 11 is the cloud mask
            seviri[11][seviri[11] == 2] = 1 #AUto einai bug giati den kanw to 1 = 0
        elif 11 in self.seviri_channels:
            # Find position of channel 11 in selected channels
            channel_11_idx = self.seviri_channels.index(11)
            seviri[channel_11_idx][seviri[channel_11_idx] == 1] = 0
            seviri[channel_11_idx][seviri[channel_11_idx] == 2] = 1

        
        # Ensure MODIS has only 1 channel (take first if multiple)
        if modis.shape[0] > 1:
            modis = modis[0:1, :, :]
        
        # Resize SEVIRI to 32x32
        if seviri.shape[1] != 32 or seviri.shape[2] != 32:
            seviri = torch.nn.functional.interpolate(
                seviri.unsqueeze(0), 
                size=(32, 32), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Ensure MODIS is 128x128
        if modis.shape[1] != 128 or modis.shape[2] != 128:
            modis = torch.nn.functional.interpolate(
                modis.unsqueeze(0),
                size=(128, 128),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Apply transforms if any
        if self.transform_lr:
            seviri = self.transform_lr(seviri)
        if self.transform_hr:
            modis = self.transform_hr(modis)
        
        # Handle NaN and Inf values
        seviri = torch.nan_to_num(seviri, nan=0.0, posinf=1.0, neginf=0.0)
        modis = torch.nan_to_num(modis, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0, 1] based on data range
        # Find min and max for normalization
        seviri_min, seviri_max = seviri.min(), seviri.max()
        modis_min, modis_max = modis.min(), modis.max()
        
        if seviri_max > seviri_min:
            seviri = (seviri - seviri_min) / (seviri_max - seviri_min)
        
        if modis_max > modis_min:
            modis = (modis - modis_min) / (modis_max - modis_min)
        
        # Clip to ensure in valid range [0, 1]
        seviri = torch.clamp(seviri, 0, 1)
        modis = torch.clamp(modis, 0, 1)
            
        return seviri, modis


# ==================== Generator (SRResNet-based) ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, input_channels=12, num_residual_blocks=16):
        super().__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual conv
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upsampling: 32x32 -> 64x64 -> 128x128 (2x upsampling twice)
        self.upsample1 = UpsampleBlock(64, scale_factor=2)
        self.upsample2 = UpsampleBlock(64, scale_factor=2)
        
        # Final output layer
        self.conv3 = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Initial feature extraction
        conv1_out = self.prelu1(self.conv1(x))
        
        # Residual blocks
        residual_out = self.residual_blocks(conv1_out)
        
        # Skip connection from initial to post-residual
        conv2_out = self.bn2(self.conv2(residual_out))
        out = conv1_out + conv2_out
        
        # Upsampling
        out = self.upsample1(out)  # 32 -> 64
        out = self.upsample2(out)  # 64 -> 128
        
        # Final convolution
        out = self.conv3(out)
        
        return torch.sigmoid(out)  # Output in range [0, 1]


# ==================== Discriminator ====================
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, stride, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, stride=1, normalize=False),
            *discriminator_block(64, 64, stride=2),
            *discriminator_block(64, 128, stride=1),
            *discriminator_block(128, 128, stride=2),
            *discriminator_block(128, 256, stride=1),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            *discriminator_block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.model(x).view(-1))


# ==================== VGG Perceptual Loss ====================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        
        # Use features before the 5th maxpool layer (layer 36)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, sr, hr):
        # VGG expects 3-channel input, so replicate the single channel
        sr_3ch = sr.repeat(1, 3, 1, 1)
        hr_3ch = hr.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(sr.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(sr.device)
        
        sr_3ch = (sr_3ch - mean) / std
        hr_3ch = (hr_3ch - mean) / std
        
        # Extract features
        sr_features = self.feature_extractor(sr_3ch)
        hr_features = self.feature_extractor(hr_3ch)
        
        return self.mse_loss(sr_features, hr_features)


# ==================== Training ====================
def train_srgan(data_dir, num_epochs=100, batch_size=16, lr_g=1e-4, lr_d=1e-4, 
                checkpoint_dir='checkpoints', device='cuda', save_images_every=5,
                seviri_channels=None, skip_black_targets=True, 
                d_train_ratio=5, label_smoothing=0.9):
    """
    Train SRGAN with improvements.
    
    Args:
        seviri_channels: List of SEVIRI channel indices to use (e.g., [0,1,2,5,8])
        skip_black_targets: Skip pairs with all-black MODIS targets
        d_train_ratio: Train discriminator every N generator updates (higher = weaker D)
        label_smoothing: Use smooth labels (0.9) instead of hard labels (1.0) for discriminator
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = SEVIRIModisDataset(data_dir, seviri_channels=seviri_channels, 
                                  skip_black_targets=skip_black_targets)
    
    # Determine number of input channels
    num_input_channels = len(seviri_channels) if seviri_channels else 12
    
    print(f"\nTotal dataset size: {len(dataset)}")
    # Split into train/validation/test (80/10/10)
    if len(dataset) < 3:
        raise ValueError(f"Dataset too small ({len(dataset)} samples). Need at least 3 samples.")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Ensure all splits have at least 1 sample
    if train_size == 0:
        train_size = 1
        val_size = len(dataset) - 1
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    if test_size == 0:
        test_size = 1
    # Adjust if total doesn't match
    total = train_size + val_size + test_size
    if total != len(dataset):
        train_size = len(dataset) - val_size - test_size
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )



    #print(f"Dataset split: {train_size} training, {val_size} validation samples")

    #train_dataset, val_dataset = torch.utils.data.random_split(
    #    dataset, [train_size, val_size]
    #)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    print(f"Teat dataset length: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize models
    generator = Generator(input_channels=num_input_channels, num_residual_blocks=16).to(device)
    discriminator = Discriminator(input_channels=1).to(device)
    
    # Loss functions
    content_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    # Optimizers - CRITICAL: Lower D learning rate to prevent collapse
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d * 0.1, betas=(0.9, 0.999))  # 10x lower!
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'val_loss': [],
        'psnr': [],
        'ssim': []
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Input channels: {num_input_channels}")
    print(f"  Discriminator train ratio: 1:{d_train_ratio}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Generator LR: {lr_g}")
    print(f"  Discriminator LR: {lr_d * 0.1}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        d_updates = 0
        
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            current_batch_size = lr_imgs.size(0)
            
            # Use label smoothing for more stable training
            real_labels = torch.ones(current_batch_size).to(device) * label_smoothing
            fake_labels = torch.zeros(current_batch_size).to(device) + (1 - label_smoothing) * 0.1
            
            # ==================== Train Discriminator (every d_train_ratio steps) ====================
            if i % d_train_ratio == 0:
                optimizer_D.zero_grad()
                
                # Real images - add noise for more robust training
                noisy_hr = hr_imgs + torch.randn_like(hr_imgs) * 0.05 * hr_imgs.std()
                noisy_hr = torch.clamp(noisy_hr, 0, 1)
                real_output = discriminator(noisy_hr)
                d_loss_real = adversarial_loss(real_output, real_labels)
                
                # Fake images
                sr_imgs = generator(lr_imgs).detach()
                noisy_sr = sr_imgs + torch.randn_like(sr_imgs) * 0.05 * sr_imgs.std()
                noisy_sr = torch.clamp(noisy_sr, 0, 1)
                fake_output = discriminator(noisy_sr)
                d_loss_fake = adversarial_loss(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                
                optimizer_D.step()
                d_updates += 1
                epoch_d_loss += d_loss.item()
            
            # ==================== Train Generator ====================
            optimizer_G.zero_grad()
            
            # Generate SR images
            sr_imgs = generator(lr_imgs)
            
            # Adversarial loss
            fake_output = discriminator(sr_imgs)
            g_adv_loss = adversarial_loss(fake_output, torch.ones(current_batch_size).to(device))
            
            # Content loss (MSE)
            g_content_loss = content_loss(sr_imgs, hr_imgs)
            
            # Perceptual loss (VGG)
            g_perceptual_loss = perceptual_loss(sr_imgs, hr_imgs)
            
            bce_loss = nn.BCELoss()
            g_bce_loss = bce_loss(sr_imgs, hr_imgs)

            # Total generator loss - adjusted weights
            g_loss = g_content_loss + 0.8 * g_bce_loss + 0.001 * g_adv_loss + 0.006 * g_perceptual_loss
            #g_loss = g_content_loss + 0.001 * g_adv_loss + 0.006 * g_perceptual_loss
            g_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            
            if (i + 1) % 100 == 0:
                avg_d = epoch_d_loss / max(d_updates, 1)
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"G_Loss: {g_loss.item():.4f}, D_Loss: {avg_d:.4f} "
                      f"(D_updates: {d_updates})")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Validation
        generator.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                sr_imgs = generator(lr_imgs)
                loss = content_loss(sr_imgs, hr_imgs)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_psnr, batch_ssim = calculate_metrics(sr_imgs, hr_imgs)
                val_psnr += batch_psnr
                val_ssim += batch_ssim
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / max(d_updates, 1)
        
        # Store history
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['val_loss'].append(avg_val_loss)
        history['psnr'].append(avg_val_psnr)
        history['ssim'].append(avg_val_ssim)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train G_Loss: {avg_g_loss:.4f}, Train D_Loss: {avg_d_loss:.4f}")
        print(f"  Val MSE Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f} dB, Val SSIM: {avg_val_ssim:.4f}")
        print(f"  D Updates: {d_updates}/{len(train_loader)}\n")
        
        # Save comparison images
        if (epoch + 1) % save_images_every == 0:
            generator.eval()
            with torch.no_grad():
                val_iter = iter(val_loader)
                lr_sample, hr_sample = next(val_iter)
                lr_sample = lr_sample.to(device)
                hr_sample = hr_sample.to(device)
                sr_sample = generator(lr_sample)
                
                save_comparison_images(lr_sample, sr_sample, hr_sample, epoch+1)
        
        # Plot training curves
        if (epoch + 1) % save_images_every == 0:
            plot_training_curves(history)
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'history': history,
                'config': {
                    'seviri_channels': seviri_channels,
                    'num_input_channels': num_input_channels,
                    'd_train_ratio': d_train_ratio,
                    'label_smoothing': label_smoothing
                }
            }, f"{checkpoint_dir}/srgan_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final training curves
    plot_training_curves(history)
        # ==================== Final Test Set Evaluation ====================
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)
    
    generator.eval()
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    test_batches = 0
    
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            sr_imgs = generator(lr_imgs)
            loss = content_loss(sr_imgs, hr_imgs)
            test_loss += loss.item()
            
            sr_imgs = (sr_imgs > 0.2).float()
            # Calculate metrics
            batch_psnr, batch_ssim = calculate_metrics(sr_imgs, hr_imgs)
            test_psnr += batch_psnr
            test_ssim += batch_ssim
            test_batches += 1
            
            # Store individual values for std calculation
            sr_np = sr_imgs.cpu().numpy()
            hr_np = hr_imgs.cpu().numpy()
            for i in range(sr_np.shape[0]):
                sr_img = np.clip(sr_np[i, 0], 0, 1)
                hr_img = np.clip(hr_np[i, 0], 0, 1)
                all_psnr.append(psnr(hr_img, sr_img, data_range=1.0))
                all_ssim.append(ssim(hr_img, sr_img, data_range=1.0))
    
    # Calculate final metrics
    final_test_loss = test_loss / test_batches
    final_test_psnr = np.mean(all_psnr)
    final_test_ssim = np.mean(all_ssim)
    std_psnr = np.std(all_psnr)
    std_ssim = np.std(all_ssim)
    
    print(f"\nTest Set Results ({len(test_dataset)} samples):")
    print(f"  MSE Loss: {final_test_loss:.6f}")
    print(f"  PSNR: {final_test_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM: {final_test_ssim:.4f} ± {std_ssim:.4f}")
    
    # Calculate bicubic baseline on test set for comparison
    print("\nComparing with Bicubic Baseline...")
    bicubic_psnr = []
    bicubic_ssim = []
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Bicubic upsampling
            lr_bicubic = torch.nn.functional.interpolate(
                lr_imgs[:, -1:, :, :], #lr_imgs[:, 0:1, :, :],  # Take first channel
                size=(128, 128),
                mode='bicubic',
                align_corners=False
            )
            
            # Calculate metrics
            lr_np = lr_bicubic.cpu().numpy()
            hr_np = hr_imgs.cpu().numpy()
            
            for i in range(lr_np.shape[0]):
                lr_img = np.clip(lr_np[i, 0], 0, 1)
                hr_img = np.clip(hr_np[i, 0], 0, 1)
                bicubic_psnr.append(psnr(hr_img, lr_img, data_range=1.0))
                bicubic_ssim.append(ssim(hr_img, lr_img, data_range=1.0))
    
    avg_bicubic_psnr = np.mean(bicubic_psnr)
    avg_bicubic_ssim = np.mean(bicubic_ssim)
    std_bicubic_psnr = np.std(bicubic_psnr)
    std_bicubic_ssim = np.std(bicubic_ssim)
    
    print(f"\nBicubic Baseline Results:")
    print(f"  PSNR: {avg_bicubic_psnr:.2f} ± {std_bicubic_psnr:.2f} dB")
    print(f"  SSIM: {avg_bicubic_ssim:.4f} ± {std_bicubic_ssim:.4f}")
    
    print(f"\nImprovement over Bicubic:")
    print(f"  ΔPSNR: +{final_test_psnr - avg_bicubic_psnr:.2f} dB")
    print(f"  ΔSSIM: +{final_test_ssim - avg_bicubic_ssim:.4f}")
    print("="*70 + "\n")
    
    # Save test results
    test_results = {
        'test_loss': final_test_loss,
        'test_psnr': final_test_psnr,
        'test_ssim': final_test_ssim,
        'test_psnr_std': std_psnr,
        'test_ssim_std': std_ssim,
        'bicubic_psnr': avg_bicubic_psnr,
        'bicubic_ssim': avg_bicubic_ssim,
        'bicubic_psnr_std': std_bicubic_psnr,
        'bicubic_ssim_std': std_bicubic_ssim,
        'improvement_psnr': final_test_psnr - avg_bicubic_psnr,
        'improvement_ssim': final_test_ssim - avg_bicubic_ssim,
        'num_test_samples': len(test_dataset)
    }
    
    # Add test results to history
    history['test_results'] = test_results
    
    # Save some test set visualizations
    with torch.no_grad():
        test_iter = iter(test_loader)
        lr_sample, hr_sample = next(test_iter)
        lr_sample = lr_sample.to(device)
        hr_sample = hr_sample.to(device)
        sr_sample = generator(lr_sample)
        
        save_comparison_images(lr_sample, sr_sample, hr_sample, 
                             epoch=num_epochs, save_dir='results/test_samples')
        print(f"Saved test set visualizations to results/test_samples/")
    
    print("Training complete!")
    return generator, discriminator, history



# ==================== Inference ====================
def inference(generator, lr_image_path, device='cuda'):
    """
    Perform super-resolution on a single LR image.
    
    Args:
        generator: Trained generator model
        lr_image_path: Path to low-resolution image
        device: Device to run inference on
    
    Returns:
        sr_image: Super-resolved image as numpy array
    """
    generator.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess image
    #import rasterio
    with rasterio.open(lr_image_path) as src:
        lr_image = src.read()  # Shape: (bands, H, W)
    
    lr_tensor = torch.from_numpy(lr_image).float().unsqueeze(0)  # Add batch dim
    
    # Resize to 32x32
    if lr_tensor.shape[2] != 32 or lr_tensor.shape[3] != 32:
        lr_tensor = torch.nn.functional.interpolate(
            lr_tensor, size=(32, 32), mode='bilinear', align_corners=False
        )
    
    # Normalize
    lr_tensor = torch.nan_to_num(lr_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    lr_min, lr_max = lr_tensor.min(), lr_tensor.max()
    if lr_max > lr_min:
        lr_tensor = (lr_tensor - lr_min) / (lr_max - lr_min)
    lr_tensor = torch.clamp(lr_tensor, 0, 1)
    
    lr_tensor = lr_tensor.to(device)
    
    # Generate SR image
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
    
    # Convert to numpy
    sr_image = sr_tensor.cpu().squeeze().numpy()
    
    return sr_image


# ==================== Main ====================
if __name__ == "__main__":
    # Configuration
    print("Starting SRGAN training...")
    DATA_DIR = "/mnt/nvme2tb/aggelos_modis_seviri_pairs/"
    CHECKPOINT_DIR = "srgan_checkpoints_v2"
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LR_G = 1e-4
    LR_D = 1e-4
    DEVICE = 'cuda'
    
    # Select specific SEVIRI channels (None = use all 12)
    # Example: Use only thermal IR channels
    SEVIRI_CHANNELS = [1,3,5,11] #None  2,4,6,12 # [3, 4, 5, 6, 7, 8, 9]  # or None for all channels
    
    # Skip black/empty targets
    SKIP_BLACK = True
    
    # Discriminator training ratio (train D every N steps)
    D_TRAIN_RATIO = 10  # Higher = weaker discriminator
    
    # Label smoothing for stability
    LABEL_SMOOTHING = 0.8 #1 shmainei oti einai sosta ta label poy anathetei o discriminator
    #1 ara ginetai pio eukola overconfident, oso to mikrainw toso pio dyskolo ginetai na ginei overconfident
    
    # Train SRGAN
    generator, discriminator, history = train_srgan(
        data_dir=DATA_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr_g=LR_G,
        lr_d=LR_D,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
        seviri_channels=SEVIRI_CHANNELS,
        skip_black_targets=SKIP_BLACK,
        d_train_ratio=D_TRAIN_RATIO,
        label_smoothing=LABEL_SMOOTHING
        )
    
    # Save final models
    torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_final.pth")
    print("Final models saved!")
        
    # Save final history
    import pickle
    with open(f"{CHECKPOINT_DIR}/training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print("Final models and training history saved!")