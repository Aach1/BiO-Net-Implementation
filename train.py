import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import BiONet

class ChestXrayMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()
        return image, mask

def dice_loss(pred, target, smooth=1e-4):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()


def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dsc = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dsc


def dice_coeff(pred, target, smooth=1e-4):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = bce_dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = bce_dice_loss(outputs, masks)
            dice = dice_coeff(outputs, masks)
            total_loss += loss.item()
            total_dice += dice.item()
    return total_loss / len(loader), total_dice / len(loader)


def main():
    train_img_dir = r"data\train_data\images"
    train_mask_dir = r"data\train_data\masks"
    val_img_dir = r"data\valid_data\images"
    val_mask_dir = r"data\valid_data\masks"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = ChestXrayMaskDataset(train_img_dir, train_mask_dir, transform)
    val_dataset = ChestXrayMaskDataset(val_img_dir, val_mask_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiONet(num_classes=1, iterations=2, num_layers=4, integrate=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_dice = 0.0
    num_epochs = 50
    save_path = "bionet_chestxray_best.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (Dice: {val_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
