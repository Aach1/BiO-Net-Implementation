# data preprocessing pipeline for Nikhil Pandey’s Chest X-ray Masks and Labels

import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = r"file/path"
IMG_DIR = os.path.join(ROOT, "CXR_png")
MASK_DIR = os.path.join(ROOT, "masks")

OUT_ROOT = r"C:\Users\Aachi\Downloads\bionet_lung"
TRAIN_IMG_OUT = os.path.join(OUT_ROOT, "train_data", "images")
TRAIN_MASK_OUT = os.path.join(OUT_ROOT, "train_data", "masks")
VAL_IMG_OUT = os.path.join(OUT_ROOT, "valid_data", "images")
VAL_MASK_OUT = os.path.join(OUT_ROOT, "valid_data", "masks")

for p in [TRAIN_IMG_OUT, TRAIN_MASK_OUT, VAL_IMG_OUT, VAL_MASK_OUT]:
    os.makedirs(p, exist_ok=True)

# 1. Collect image–mask pairs
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
pairs = []

for img_path in img_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    # expected mask file names: CHNCXR_0001_0_mask.png, MCUCXR_0399_1.png, etc.
    mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")
    if not os.path.exists(mask_path):
        alt = os.path.join(MASK_DIR, f"{base}.png")
        if os.path.exists(alt):
            mask_path = alt
        else:
            continue
    pairs.append((img_path, mask_path))

print(f"Found {len(pairs)} image–mask pairs.")

# 2. Train–validation split
train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
print(f"Training: {len(train_pairs)} | Validation: {len(val_pairs)}")


# 3. Image processing helper
def process_and_save(img_path, mask_path, out_img_path, out_mask_path, size=(512, 512)):
    # open image & mask
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = img.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)

    mask = mask.point(lambda p: 255 if p > 127 else 0).convert("L")

    img.save(out_img_path)
    mask.save(out_mask_path)


# 4. Process and export all pairs
def save_pairs(pairs, out_img_dir, out_mask_dir, subset_name):
    for i, (img_p, mask_p) in enumerate(pairs):
        base = os.path.splitext(os.path.basename(img_p))[0]
        out_img = os.path.join(out_img_dir, f"{base}.png")
        out_mask = os.path.join(out_mask_dir, f"{base}.png")
        process_and_save(img_p, mask_p, out_img, out_mask)
    print(f"{subset_name} set saved: {len(pairs)} samples.")


save_pairs(train_pairs, TRAIN_IMG_OUT, TRAIN_MASK_OUT, "Train")
save_pairs(val_pairs, VAL_IMG_OUT, VAL_MASK_OUT, "Validation")

print("\nPreprocessing complete!")
print(f"Data ready at: {OUT_ROOT}")
