import torch
import numpy as np
import torch.nn.functional as F

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
  def __init__(self, upscale_factor):
    self.upscale_factor = upscale_factor
  def __call__(self, image):

    image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3], order='F')
    image = image[:,:,:,np.newaxis] # add in a dimension for channel
    image = image.transpose((2,3,0,1)) # change order of axis to (batch_size * channel * width * height)
    image = torch.from_numpy(image)
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
  # def __init__(self, opt):
  #   self.upscale_factor = opt["upscale_factor"]

  def __call__(self, label):

    label = torch.from_numpy(label).float()
    class_values = [1,2,3] # used instead of unique for speed
    label = label.unsqueeze(0).repeat(3,1,1,1)
#    label = F.interpolate(label, scale_factor=1/self.upscale_factor, mode='bilinear')

    for dimension, value in enumerate(class_values):
      label_channel = torch.where(label[dimension,:,:,:] >= value, label[dimension,:,:,:], torch.tensor(0.))
      label_channel = torch.where(label[dimension,:,:,:] < value, label_channel, torch.tensor(1.))
      label[dimension,:,:,:] = label_channel
    label = label.repeat(1,1,1,4).permute(3,0,1,2)
    return label
