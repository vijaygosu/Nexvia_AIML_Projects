from torchvision import transforms as T

from medimg.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def eval_transform(size: int = IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)), T.Grayscale(3), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(size: int = IMG_SIZE):
    # No horizontal flip: it would create anatomically implausible (dextrocardia-like) images.
    return T.Compose([
        T.Grayscale(3),
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomRotation(8),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
