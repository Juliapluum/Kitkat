from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.datasets import ImageFolder

class AlbumentationsImageFolder(ImageFolder):

    def __init__(self,root,transform=None):
        super().__init__(root=root,transform=None)

        self.albumentations_transform = transform


    def __getitem__(self, index):

        path, target = self.samples[index]

        image = Image.open(path).convert("RGB")

        image = np.array(image)


        if self.albumentations_transform is not None:

            augmented = self.albumentations_transform(
                image=image
            )

            image = augmented["image"]


        return image, target
