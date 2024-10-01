from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


SIZE = 224

class WNetData(Dataset):
    def __init__(self, data_dir, mode, input_transforms):
        self.mode = mode
        self.data_path  = os.path.join(data_dir, mode)
        self.images_dir = os.path.join(self.data_path, 'images')
        self.image_list = self.get_image_list()
        self.transforms = input_transforms

        self.randomCrop = transforms.RandomCrop(SIZE)
        self.centerCrop = transforms.CenterCrop(SIZE)
        self.toTensor   = transforms.ToTensor()
        self.toPIL      = transforms.ToPILImage()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        filepath = self.image_list[i]
        input = self.load_pil_image(filepath)
        input = self.transforms(input)

        input = self.toPIL(input)
        output = input.copy()
        if self.mode == "train":
            output = self.randomCrop(input)
        input = self.toTensor(self.centerCrop(input))
        output = self.toTensor(output)

        return input, output

    def get_image_list(self):
        image_list = []
        for file in os.listdir(self.images_dir):
            path = os.path.join(self.images_dir, file)
            image_list.append(path)
        return image_list

    def load_pil_image(self, path):
        img = Image.open(path)
        return img.convert('RGB')