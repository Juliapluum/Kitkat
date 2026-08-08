import torch

import torchvision.transforms as transforms

import random




class RandomDarkness:

    def __init__(self, p=0.5, min_factor=0.55, max_factor=0.80):
        self.p = p
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):

        if random.random() > self.p:
            return img

        factor = random.uniform(
            self.min_factor,
            self.max_factor
        )

        return transforms.functional.adjust_brightness(
            img,
            factor
        )


class RandomNight:

    def __init__(
        self,
        p=0.35,
        brightness_min=0.20,
        brightness_max=0.40
    ):
        self.p = p
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def __call__(self, img):

        if random.random() > self.p:
            return img

        brightness = random.uniform(
            self.brightness_min,
            self.brightness_max
        )

        img = transforms.functional.adjust_brightness(
            img,
            brightness
        )

        tensor = transforms.functional.pil_to_tensor(img).float() / 255.0

        tensor[2] = torch.clamp(
            tensor[2] + 0.08,
            0.0,
            1.0
        )

        return transforms.functional.to_pil_image(tensor)

class RandomBlur:

    def __init__(
        self,
        p=0.45,
        kernel_min=3,
        kernel_max=9
    ):
        self.p = p
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max

    def __call__(self, img):

        if random.random() > self.p:
            return img

        kernel_size = random.choice(
            [3, 5, 7, 9]
        )

        sigma = random.uniform(
            0.5,
            2.0
        )

        return transforms.functional.gaussian_blur(
            img,
            kernel_size=kernel_size,
            sigma=sigma
        )

random_occlusion = transforms.RandomErasing(
    p=0.30,
    scale=(0.05, 0.30),
    ratio=(0.5, 2.0),
    value=0
)
