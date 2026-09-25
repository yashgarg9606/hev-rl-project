import sys
import torch
import numpy
import pandas
import scipy
import matplotlib
import gymnasium

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())