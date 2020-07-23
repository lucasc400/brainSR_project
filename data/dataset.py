import torch.utils.data as data
from data.transforms import input_transform, target_transform
import numpy as np

class brain_tumour_dataset(data.Dataset):
    def __init__(self, opt, image_list):
        super(brain_tumour_dataset, self).__init__()

        self.image_list = image_list
        self.upscale_factor = opt["upscale_factor"]

        self.input_transform = input_transform(upscale_factor=self.upscale_factor)
        self.target_transform = target_transform()

        self.scale = opt["scale"]
        # training
        self.use_shuffle = opt.get("use_shuffle")
        self.batch_size = opt.get("batch_size") if opt.get("batch_size") else None
        # valid
        self.depth_padding = opt.get("depth_padding")

    def __getitem__(self, index):
        input = np.asanyarray(self.image_list[index].dataobj)

        # scale
        if self.scale:
            input = input / np.amax(input, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]

        # Padding
        if self.depth_padding:
            input = np.pad(input, ((0, 0), (0, 0), (0, self.depth_padding), (0, 0)))  # zero padding

        target = input.copy()

        # Transform
        input = self.input_transform(input)
        target = self.target_transform(target)

        # batch (training)
        if self.batch_size:
            batch_index = np.random.randint(0, input.shape[0], self.batch_size)
            input = input[batch_index, :, :, :]
            target = target[batch_index, :, :, :]

        return dict(L=input, H=target)

    def __len__(self):
        return len(self.image_list)


def create_dataset(opt_dataset, image_list):
    return brain_tumour_dataset(opt_dataset, image_list)
