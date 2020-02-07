import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, 'shadow_train') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'shadow_free') + '/*.*'))

    def image_open(self, name):
        image = Image.open(name)
        if image.mode == 'RGBA':
            image2 = Image.new('RGB', image.size, (255, 255, 255))
            image2.paste(image, mask=image.split()[3])
            image = image2
        return image

    def __getitem__(self, index):
        item_A = self.transform(self.image_open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(self.image_open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(self.image_open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
