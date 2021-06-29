# Adding the file main.py for executing the code. Will consist of methods for training and testing the neural network

from vision.nn import UNet
import torch

if __name__ == '__main__':
  net = UNet()
  image = torch.rand((1,1,572,572))
  net.forward(image) # Passing the random tensor in place of the image for testing
  
