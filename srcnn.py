#!/usr/bin/env python3
"""
SEVIRI → MODIS Super-Resolution (SRCNN)
Input: SEVIRI spectral bands (11, 32×32)
Target: MODIS cloud mask (1, 128×128)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ======================
# Dataset
# ======================
'''class PairedSeviriModisDataset(Dataset):
    def __init__(self, pairs, modis_suffix="_modis.tif", seviri_suffix="_seviri.tif",
                 target_size=128, input_size=32, channels=None,
                 transform=None, skip_if_empty=True):
        self.pairs = pairs
        self.modis_suffix = modis_suffix
        self.seviri_suffix = seviri_suffix
        self.target_size = target_size
        self.input_size = input_size
        self.transform = transform
        self.skip_if_empty = skip_if_empty
        self.channels = channels if channels is not None else list(range(1, 12))  # default: 11 chans

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem = self.pairs[idx]
        m_file = Path(stem + self.modis_suffix)
        s_file = Path(stem + self.seviri_suffix)

        # MODIS (always 128×128, 1 band)
        with rasterio.open(m_file) as mds:
            m = mds.read(out_shape=(1, self.target_size, self.target_size)).astype("float32")
        m = np.nan_to_num(m, nan=0.0)

        # SEVIRI (12 bands)
        with rasterio.open(s_file) as sds:
            s = sds.read(out_shape=(sds.count, self.target_size, self.target_size)).astype("float32")
        s = np.nan_to_num(s, nan=0.0)

        sel = [c - 1 for c in self.channels]  # convert 1-based to 0-based
        s = s[sel]

        # Downsample to input_size
        s_small = np.zeros((s.shape[0], self.input_size, self.input_size), dtype=np.float32)
        factor = self.target_size // self.input_size
        for b in range(s.shape[0]):
            for y in range(self.input_size):
                for x in range(self.input_size):
                    y0, y1 = y * factor, (y + 1) * factor
                    x0, x1 = x * factor, (x + 1) * factor
                    s_small[b, y, x] = s[b, y0:y1, x0:x1].mean()

        if self.skip_if_empty and (not np.any(m) or not np.any(s_small)):
            return self.__getitem__((idx + 1) % len(self))

        return torch.from_numpy(s_small).float(), torch.from_numpy(m).float()'''
class PairedSeviriModisDataset(Dataset):
    def __init__(self, pairs, modis_suffix="_modis.tif", seviri_suffix="_seviri.tif",
                 target_size=128, input_size=32, channels=None,
                 transform=None, skip_if_empty=True):
        """
        Args:
            pairs: list of stems (without suffix)
            channels: list of SEVIRI channel indices to keep (1-based). 
                      Default = first 11 (exclude cloud mask).
        """
        self.pairs = pairs
        self.modis_suffix = modis_suffix
        self.seviri_suffix = seviri_suffix
        self.target_size = target_size
        self.input_size = input_size
        self.transform = transform
        self.skip_if_empty = skip_if_empty

        # Default: use first 11 SEVIRI channels
        self.channels = channels if channels is not None else list(range(1, 12))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem = self.pairs[idx]
        m_file = Path(stem + self.modis_suffix)
        s_file = Path(stem + self.seviri_suffix)

        # --- MODIS (always 128×128, 1 band) ---
        with rasterio.open(m_file) as mds:
            m = mds.read(out_shape=(1, self.target_size, self.target_size)).astype("float32")
        m = np.nan_to_num(m, nan=0.0)

        # Skip black MODIS patches
        if self.skip_if_empty and not np.any(m):
            return self.__getitem__((idx + 1) % len(self))

        # --- SEVIRI (12 bands) ---
        with rasterio.open(s_file) as sds:
            s = sds.read(out_shape=(sds.count, self.target_size, self.target_size)).astype("float32")
        s = np.nan_to_num(s, nan=0.0)

        # Select requested channels (convert to 0-based index for numpy)
        sel = [c - 1 for c in self.channels]
        s = s[sel]

        # --- Handle SEVIRI channel 12 cloud mask ---
        # If channel 12 is selected, remap values {0,1,2} -> {0,1,0}
        if 12 in self.channels:
            idx12 = self.channels.index(12)  # position in selected channels
            cmask = s[idx12]
            cmask_bin = np.where(cmask == 2, 1.0, 0.0).astype("float32")
            s[idx12] = cmask_bin

        # Downsample to input_size
        s_small = np.zeros((s.shape[0], self.input_size, self.input_size), dtype=np.float32)
        factor = self.target_size // self.input_size
        for b in range(s.shape[0]):
            for y in range(self.input_size):
                for x in range(self.input_size):
                    y0, y1 = y * factor, (y + 1) * factor
                    x0, x1 = x * factor, (x + 1) * factor
                    s_small[b, y, x] = s[b, y0:y1, x0:x1].mean()

        # Skip if SEVIRI patch is empty
        if self.skip_if_empty and not np.any(s_small):
            return self.__getitem__((idx + 1) % len(self))

        return torch.from_numpy(s_small).float(), torch.from_numpy(m).float()

# ======================
# Model (SRCNN-like)
# ======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class SpatialSRCNN(nn.Module):
    def __init__(self, input_channels=11):
        super().__init__()
        self.initial_up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.residuals = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 32),
        )
        self.final_up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_up(x)        # 32→64
        x = self.feature(x)
        x = self.residuals(x)
        x = self.final_up(x)          # 64→128
        return self.reconstruction(x) # → (1,128,128)

# ======================
# Training & Evaluation
# ======================
def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for s, m in loader:
        s, m = s.to(device), m.to(device)
        optimizer.zero_grad()
        pred = model(s)
        loss = criterion(pred, m)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * s.size(0)
    return total_loss / len(loader.dataset)

'''def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr, total_ssim = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for s, m in loader:
            s, m = s.to(device), m.to(device)
            pred = model(s)
            loss = criterion(pred, m)
            total_loss += loss.item() * s.size(0)

            sr_np, hr_np = pred.cpu().numpy(), m.cpu().numpy()
            for i in range(sr_np.shape[0]):
                pred_im = np.clip(sr_np[i, 0], 0, 1)
                hr_im   = np.clip(hr_np[i, 0], 0, 1)
                total_psnr += psnr(hr_im, pred_im, data_range=1.0)
                total_ssim += ssim(hr_im, pred_im, data_range=1.0)
            count += sr_np.shape[0]

    return (total_loss / len(loader.dataset),
            total_psnr / count,
            total_ssim / count)
'''
def iou_score(pred, target):
    intersection = np.logical_and(target == 1, pred == 1).sum()
    union = np.logical_or(target == 1, pred == 1).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_score(pred, target):
    intersection = np.logical_and(target == 1, pred == 1).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)

def evaluate(loader, model, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_psnr, total_ssim = 0.0, 0.0
    total_iou, total_dice = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            # Loss on probabilities
            loss = criterion(sr, hr)
            total_loss += loss.item() * lr.size(0)

            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()

            for i in range(sr_np.shape[0]):
                pred = (sr_np[i, 0]) #> threshold).astype(np.float32)
                gt   = hr_np[i, 0]

                total_psnr += psnr(gt, pred, data_range=1.0)
                total_ssim += ssim(gt, pred, data_range=1.0)
                total_iou  += iou_score(pred, gt)
                total_dice += dice_score(pred, gt)

            count += sr_np.shape[0]

    avg_loss = total_loss / len(loader.dataset)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_iou  = total_iou / count
    avg_dice = total_dice / count
    return avg_loss, avg_psnr, avg_ssim, avg_iou, avg_dice

def save_examples(model, loader, device, out_dir="examples", max_batches=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (s, m) in enumerate(loader):
            if batch_idx >= max_batches: break
            s, m = s.to(device), m.to(device)
            pred = model(s)
            s, m, pred = s.cpu().numpy(), m.cpu().numpy(), pred.cpu().numpy()
            for i in range(s.shape[0]):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(s[i, -1], cmap="gray")
                axs[0].set_title("SEVIRI (32×32, ch1)")
                axs[1].imshow(m[i, 0], cmap="gray")
                axs[1].set_title("MODIS (128×128)")
                axs[2].imshow(pred[i, 0], cmap="gray")
                axs[2].set_title("Super-Resolved")
                for ax in axs: ax.axis("off")
                plt.tight_layout()
                plt.savefig(out_dir / f"example_{batch_idx}_{i}.png")
                plt.close()

def plot_curves(train_losses, val_losses, val_psnrs, val_ssims, out_path="training_curves.png"):
    epochs = range(1, len(train_losses)+1)
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(epochs, val_psnrs, label="PSNR")
    axs[1].set_title("Validation PSNR")
    axs[2].plot(epochs, val_ssims, label="SSIM")
    axs[2].set_title("Validation SSIM")
    for ax in axs: ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ======================
# Main
# ======================
def main():
    pairs_dir = Path("/mnt/nvme2tb/aggelos_modis_seviri_pairs")
    modis_files = sorted(pairs_dir.glob("*_modis.tif"))
    stems = [str(f).replace("_modis.tif", "") for f in modis_files]

    train_stems, temp_stems = train_test_split(stems, test_size=0.2, random_state=42)
    val_stems, test_stems = train_test_split(temp_stems, test_size=0.5, random_state=42)

    train_ds = PairedSeviriModisDataset(train_stems, channels=[2,4,6,12])
    val_ds   = PairedSeviriModisDataset(val_stems, channels=[2,4,6,12])
    test_ds  = PairedSeviriModisDataset(test_stems, channels=[2,4,6,12])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialSRCNN(input_channels=len(train_ds.channels)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    best_val_loss = float("inf")
    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []

    num_epochs = 10
    for epoch in range(num_epochs):
        tr_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        #val_loss, val_psnr, val_ssim = evaluate(val_loader, model, criterion, device)
        val_loss, val_psnr, val_ssim, val_iou, val_dice = evaluate(val_loader, model, criterion, device)

        print(f"Epoch {epoch+1:02d}/{num_epochs} "
            f"- Train loss: {tr_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.3f}, "
            f"IoU: {val_iou:.3f}, Dice: {val_dice:.3f}")


        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        #print(f"Epoch {epoch+1:03d}/{num_epochs} - "
        #      f"Train_loss: {tr_loss:.4f} | "
        #      f"Val_loss: {val_loss:.4f} | "
        #      f"val_PSNR: {val_psnr:.2f} | "
        #      f"val_SSIM: {val_ssim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_srcnn.pth")

    plot_curves(train_losses, val_losses, val_psnrs, val_ssims)
    save_examples(model, test_loader, device)

if __name__ == "__main__":
    main()