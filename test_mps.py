import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

x = torch.rand(2000, 2000, device=device)
y = torch.rand(2000, 2000, device=device)

z = x @ y

print("Computation successful!")
print("Result shape:", z.shape)