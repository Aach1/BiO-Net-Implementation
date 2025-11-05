import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import imgaug.augmenters as iaa
import imgaug.augmentables.segmaps as iaa_segmaps

class BiONetDataset(Dataset):

    def __init__(self, path, data_name='chestxray', batchsize=1, steps=None, shuffle=False, transforms=None, augment=True):
        self.data_name = data_name
        self.x, self.y = self.load_data(path, data_name)
        self.steps = steps
        self.shuffle = shuffle

        if steps is not None:
            self.idx_mapping = np.random.randint(0, self.x.shape[0], steps * batchsize)
            self.steps = self.steps * batchsize

        # If transforms provided externally, use them
        if transforms is not None:
            self.transforms = transforms
        elif augment:
            self.transforms = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.3),
    iaa.Affine(rotate=(-25, 25), scale=(0.9, 1.1)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
])

        else:
            self.transforms = None


    def __len__(self):
        return self.steps if self.steps is not None else self.x.shape[0]
    
    def __getitem__(self, index):
        if self.steps is not None:
            index = self.idx_mapping[index]

        image = self.x[index]
        mask = self.y[index]

        if self.transforms is not None:
            segmap = iaa_segmaps.SegmentationMapsOnImage(mask, shape=image.shape)
            augmented = self.transforms(image=image, segmentation_maps=segmap)
            image = augmented[0]
            mask = augmented[1].get_arr()

        image = image.astype('float32') / 255.0
        mask = (mask > 127).astype('float32')

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        image = ToTensor()(image)
        mask = ToTensor()(mask)

        return image, mask


    def load_data(self, path, data_name):
        """
        Loads image and mask data for the given dataset.
        """
        print(f'Loading {data_name} data from {path}...')

        if data_name == 'chestxray':
            img_dir = os.path.join(path, 'images')
            mask_dir = os.path.join(path, 'masks')
            img_files = sorted(os.listdir(img_dir))
            mask_files = sorted(os.listdir(mask_dir))

            imgs_list = []
            masks_list = []

            for img_name, mask_name in zip(img_files, mask_files):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)

                img = np.array(Image.open(img_path).convert('L'))
                mask = np.array(Image.open(mask_path).convert('L'))

                imgs_list.append(img)
                masks_list.append(mask)

        else:
            raise ValueError(f"Unsupported dataset: {data_name}")

        imgs_np = np.asarray(imgs_list, dtype=np.uint8)
        masks_np = np.asarray(masks_list, dtype=np.uint8)

        print(f"Loaded {len(imgs_np)} samples from {path}")
        print("Images shape:", imgs_np.shape, "Masks shape:", masks_np.shape)
        
        return imgs_np, masks_np
