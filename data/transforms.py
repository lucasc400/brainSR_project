import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from data.gaussian import GaussianSmoothing

class input_transform(object):
  '''
  Args:
    Upscale factor
  Returns:
    A function that transforms a numpy array of shape (width * height * depth * MRI_parameters)
    To a pytorch tensor with shape (batch_size * channel * width * height)
    Where batch_size = depth * MRI_parameters, and width and height are downsampled by upscale
    factor, using bilinear interpolation
  '''
  def __init__(self, upscale_factor, gaussian_opt):
    self.upscale_factor = upscale_factor
    if gaussian_opt is not None:
        self.gaussian = GaussianSmoothing(channels=1, kernel_size=gaussian_opt["kernel_size"], sigma=gaussian_opt["sigma"], dim=gaussian_opt["dim"])
    elif gaussian_opt is None:
        self.gaussian = None
    
  def __call__(self, image):

    image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3], order='F')
    image = image[:,:,:,np.newaxis] # add in a dimension for channel
    image = image.transpose((2,3,0,1)) # change order of axis to (batch_size * channel * width * height)
    image = torch.from_numpy(image)
    if self.gaussian is not None:
        image = self.gaussian(image)
    image = F.interpolate(image, size=(int(image.shape[2]/self.upscale_factor), int(image.shape[3]/self.upscale_factor)), mode='bilinear')
    return image


class target_transform(object):
  '''
  Returns:
    A function that transforms a numpy array of shape (width * height * depth * MRI_parameters)
    To a pytorch tensor with shape (batch_size * channel * width * height)
  '''
  def __call__(self, image):
    image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3], order='F')
    image = image[:,:,:,np.newaxis] # add in a dimension for channel
    image = image.transpose((2,3,0,1)) # change order of axis to (batch_size * channel * width * height)
    image = torch.from_numpy(image)
    return image

class label_transform(object):
  '''
  For a label with shape (155, 240, 240) with 3 classes,
  Return a tensor of shape (620, 3, 240, 240) where each channel maps a class
  '''
  def __init__(self, opt):
    self.upscale_factor = opt["upscale_factor"]
    self.class_values = opt.get("class_values")
    if self.class_values is None:
        self.class_values = [1., 2., 3.]

  def __call__(self, label):

    label = torch.from_numpy(label).float()
    label = label.unsqueeze(0).repeat(len(self.class_values),1,1,1)
#    label = F.interpolate(label, scale_factor=1/self.upscale_factor, mode='bilinear')

    for dimension, value in enumerate(self.class_values):
      label_channel = torch.where(label[dimension,:,:,:] >= value, label[dimension,:,:,:], torch.tensor(0.))
      label_channel = torch.where(label[dimension,:,:,:] < value, label_channel, torch.tensor(1.))
      label[dimension,:,:,:] = label_channel
    label = label.repeat(1,1,1,4).permute(3,0,1,2)
    return label

class centres_list_generator(object):
  def __init__(self, patch_size, image_shape, batch_size):
    dim0, dim1, dim2, dim3 = image_shape
    patch_size = int(patch_size/2)
    self.ranges = np.array([[patch_size, dim0-patch_size],
                       [patch_size, dim1-patch_size],
                       [patch_size, dim2-patch_size],
                       [0, dim3]])
    self.batch_size = batch_size
  def __call__(self):
    return np.random.randint(self.ranges[:,0], self.ranges[:,1], size=(self.batch_size, self.ranges.shape[0]))
    
class create_batches_with_patches(object):
  def __init__(self, batch_size, patch_size=64):
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.half_patch_size = int(patch_size/2)

  def __call__(self, image, centres):
    batch = np.zeros((self.batch_size, 1, self.patch_size, self.patch_size, self.patch_size))
    for i in range(self.batch_size):
      centre = centres[i]
      batch[i,0,:,:,:] = image[centre[0] - self.half_patch_size : centre[0] + self.half_patch_size,
                               centre[1] - self.half_patch_size : centre[1] + self.half_patch_size, 
                               centre[2] - self.half_patch_size : centre[2] + self.half_patch_size, 
                               centre[3]]
    batch = torch.from_numpy(batch).float()
    return batch

class input_transform_3D(object):
  '''
  Input: 5D images of shape (batch, channel, depth, width, height)
  Output: Gaussian + Interpolation
  '''
  def __init__(self, upscale_factor, gaussian_opt):
    self.upscale_factor = upscale_factor
    if gaussian_opt is not None:
      self.gaussian = GaussianSmoothing(channels=1, 
                                        kernel_size=gaussian_opt["kernel_size"], 
                                        sigma=gaussian_opt["sigma"],
                                        dim=gaussian_opt["dim"])
    elif gaussian_opt is None:
      self.gaussian = None
    
  def __call__(self, image):
    image = self.gaussian(image)
    image = F.interpolate(image, scale_factor=1/4, mode='trilinear')
    return image

class target_transform_3D(object):
  '''
  Gaussian
  '''
  def __init__(self, gaussian_opt):
    self.gaussian = GaussianSmoothing(channels=1, 
                                      kernel_size=gaussian_opt["kernel_size"], 
                                      sigma=gaussian_opt["sigma"],
                                      dim=gaussian_opt["dim"])
  def __call__(self, image):
    return self.gaussian(image)