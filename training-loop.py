import torch 
import numpy as np 


# ---------- TENSORS –––––––––––––––––––––––––––––––––––––
# tensors are similar to numpy arrays and matrices but they are optimised for automatic differentiation 
# and are optmised to run on GPUs or other hardware accelerators.

# --------------- Initialising a Tensor --------------------------
# Tensors can be initialised in various ways: 

# - Directly from data - Tensors can be created directly form data. The data type is automatically inferred 

data = [[1,2],[3,4]]
x_data = torch.tensor(data) # these are like different torch methods based on the input


# - From a Numpy array - Tensors can be created from NumpPy arrays (and vice versa)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# - From another tensor
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overriden. 
x_ones = torch.ones_like(x_data) # retains the properties of x_data 
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype = torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n" )

# - With random or constant values: 
# shape is a tuple of tensor dimensions. 
# In the functions below, it determines the dimensions of the output tensor 

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor : \n {ones_tensor} \n")
print(f"Zeros Tensor : \n {zeros_tensor} \n")

# -------------- Attributes of a Tensor --------------------------------------

# Tensors attributes describe their shape, datatype and the device on which they are stored

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# -------------- Operations on Tensors ------------------------------------------



# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

# Standard numpy-like indexing and slicing: 
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors: You can use torch.cat to concatenate a sequence of tensors along a given dimension. 
# See also torch.stack, another tensor joining operator that is subtly different from torch.cat

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

# Arithmetic operations 
    # Matrix multiplication between two tensors 
    # Element wise product 

# Single-element tensors 
    # Converting one element tensor to a python numerical value 
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations

# --------- DATASETS & DATALOADERS ----------------------------------

# It's ideal to decouple dataset code and data processing code from model training code for better readability and modularity 
# PyTorch provides two data primitives 
    # torch.utils.data.DataLoader 
    # torch.utils.utils.Dataset 

# Dataset stores the samples and their corresponding labels
# DataLoader wraps an iterable around Dataset to enable easy access to the samples 

# Pre-loaded datasets: Image, Datasets, Text Datasets, Audio Datasets 

# ------------- Loading a Dataset -------------------------------------

# Example using Fashion-MNIST - from TorchVision 
# 60,000 training examples 10,000 test examples 
# associated label from one of 10 classes 

# load FashionMNIST dataset with the following parameters: 
# root - the path where the train/ test data is stored , 
# train - specifies training or test dataset,
# download=True - downloads the data from the internet if it's not available at root,
# transform, target_transform specify the feature and label transformations

import torch 
from torch.utils.data import Dataset 
from torchvision import datasets 
from torchvision.transforms import v2 
import matplotlib.pyplot as plt 

training_data = datasets.FashionMNIST(
    root = "data", 
    train = True, 
    download = True, 
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False, 
    download = True, 
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

# ----------- Iterating and Visualising the Dataset --------------------------------


# ---------- Creating a Custom Dataset for your files ------------------------------

# ------------ Preparing your data for training with DataLoaders -------------------

# -------------- BUILD THE NEURAL NETWORK ------------------------------------------

import os 
import torch 
from torch import nn 
from torch.uitls.data import DataLoader 
from torchvision import datasets, transforms 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        ) 

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 

model = NeuralNetwork().to(device)
print(model)

