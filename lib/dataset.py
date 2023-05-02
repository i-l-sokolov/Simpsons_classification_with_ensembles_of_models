import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def get_augmentation():
    train_transforms = transforms.Compose(nn.ModuleList([
        transforms.RandomResizedCrop(224, scale=(0.45, 0.9)),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomAutocontrast(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]))
    return train_transforms


def get_valid_trans():
    valid_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return valid_transform


def restore_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[0] = img[0] * std[0] + mean[0]
    img[1] = img[1] * std[1] + mean[1]
    img[2] = img[2] * std[2] + mean[2]
    img = Image.fromarray((np.moveaxis(img.numpy(), 0, -1) * 255).astype(np.uint8))
    return img


class SimpsonsDataset2(Dataset):

    def __init__(self, images, labels, labels_codes, probs, mode):
        super(SimpsonsDataset2, self).__init__()
        self.images = images
        self.labels = labels
        self.label_codes = labels_codes
        self.mode = mode
        self.probs = probs
        self.augmentation = get_augmentation()
        self.valid_transforms = get_valid_trans()

    def prepare_image(self, image):
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float64) / 255
        image = self.valid_transforms(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image.load()
        if self.mode == 'train':
            label = self.label_codes[self.labels[idx]]
            prob = self.probs[label]
            image = self.augmentation(image)
            image = self.valid_transforms(image)
            ret = (image, torch.tensor(label))
        elif self.mode == 'valid':
            image = self.valid_transforms(image)
            label = torch.tensor(self.label_codes[self.labels[idx]])
            ret = (image, label)
        elif self.mode == 'test':
            image = self.valid_transforms(image)
            ret = image
        return ret
