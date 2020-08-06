import torch.utils.data as data
from data.transforms import input_transform, target_transform, label_transform # for 2D
from data.transforms import centres_list_generator, create_batches_with_patches, input_transform_3D, target_transform_3D # for 3D
import numpy as np

class brain_tumour_dataset(data.Dataset):
    def __init__(self, opt, image_list, use_condition=False):
        super(brain_tumour_dataset, self).__init__()
        # data
        self.image_list = image_list
        self.use_condition = use_condition

        # transform
        self.upscale_factor = opt["upscale_factor"]
        self.gaussian = opt.get("gaussian")
        self.input_transform = input_transform(upscale_factor=self.upscale_factor, gaussian_opt=self.gaussian)
        self.target_transform = target_transform()
        self.label_transform = label_transform(opt)
        self.scale = opt["scale"]

        # training
        self.use_shuffle = opt.get("use_shuffle")
        self.batch_size = opt.get("batch_size") if opt.get("batch_size") else None

        # valid
        self.depth_padding = opt.get("depth_padding")

    def __getitem__(self, index):
        if self.use_condition is True:
            input = np.asanyarray(self.image_list[index].get("image").dataobj)
            label = np.asanyarray(self.image_list[index].get("label").dataobj)
        else:
            input = np.asanyarray(self.image_list[index].dataobj)

        # scale
        if self.scale:
            input = input / np.amax(input, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]

        # Padding
        if self.depth_padding:
            input = np.pad(input, ((0, 0), (0, 0), (0, self.depth_padding), (0, 0)))  # zero padding
            if self.use_condition is True:
                if label is not None:
                    label = np.pad(label, ((0, 0), (0, 0), (0, self.depth_padding)))
        target = input.copy()

        # Transform
        input = self.input_transform(input)
        target = self.target_transform(target)
        if self.use_condition is True:
            if label is not None:
                label = self.label_transform(label)

        # batch (training)
        if self.batch_size:
            batch_index = np.random.randint(0, input.shape[0], self.batch_size)
            input = input[batch_index, :, :, :]
            target = target[batch_index, :, :, :]
            if self.use_condition is True and label is not None:
                label = label[batch_index, :, :, :]
        
        if self.use_condition is True and label is not None:
            target = (target, label)
        
        return dict(L=input, H=target)

    def __len__(self):
        return len(self.image_list)

class brain_tumour_dataset_3D(data.Dataset):
    def __init__(self, opt, image_list, use_condition=False):
        super(brain_tumour_dataset_3D, self).__init__()
        # data
        self.image_list = image_list
        self.use_condition = use_condition

        # transform
        self.upscale_factor = opt["upscale_factor"]
        self.gaussian = opt.get("gaussian")
        self.input_transform = input_transform_3D(upscale_factor=self.upscale_factor, gaussian_opt=self.gaussian)
        self.target_transform = target_transform_3D(gaussian_opt=self.gaussian)
        # self.label_transform = label_transform(opt)
        self.scale = opt["scale"]

        # training
        self.use_shuffle = opt.get("use_shuffle")
        self.batch_size = opt.get("batch_size") if opt.get("batch_size") else None

        # valid
        self.depth_padding = opt.get("depth_padding")

        # patch_generator
        self.use_patch = opt.get('use_patch')
        if self.use_patch:
          self.patch_size = opt.get("patch_size")
          self.generate_centres = centres_list_generator(opt["patch_size"], opt["image_shape"], opt["batch_size"])
          self.create_batches = create_batches_with_patches(self.batch_size, self.patch_size)

    def __getitem__(self, index):
        # if self.use_condition is True:
        #     input = np.asanyarray(self.image_list[index].get("image").dataobj)
        #     label = np.asanyarray(self.image_list[index].get("label").dataobj)
        # else:
        #     input = np.asanyarray(self.image_list[index].dataobj)
        input = np.asanyarray(self.image_list[index].dataobj)

        # scale
        if self.scale:
            input = input / np.amax(input, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]

        # Padding
        if self.depth_padding:
            input = np.pad(input, ((0, 0), (0, 0), (0, self.depth_padding), (0, 0)))  # zero padding
            # if self.use_condition is True:
            #     if label is not None:
            #         label = np.pad(label, ((0, 0), (0, 0), (0, self.depth_padding)))
        target = input.copy()
        
        # Generate batches and 3D patches
        if self.use_patch:
            centres = self.generate_centres()
            input = self.create_batches(input, centres)
            target = input.clone()
        else:
            input = input.transpose((3,2,0,1))[:,np.newaxis,:,:,:]
            input = torch.from_numpy(input)
            target = input.clone()
            
        # Transform
        input = self.input_transform(input)
        target = self.target_transform(target)
        # if self.use_condition is True:
        #     if label is not None:
        #         label = self.label_transform(label)

        # if self.use_condition is True and label is not None:
        #     target = (target, label)
        
        return dict(L=input, H=target)

    def __len__(self):
        return len(self.image_list)


def create_dataset(opt_dataset, image_list, use_condition=None, use_3D=False):
    if use_3D is False:
        return brain_tumour_dataset(opt_dataset, image_list, use_condition)
    
    elif use_3D is True:
        return brain_tumour_dataset_3D(opt_dataset, image_list, use_condition=False)