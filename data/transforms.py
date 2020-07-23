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
