import torch

# Tensors are just like arrays
x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

# Array math
print(x*y)

# Initialize a new array
x = torch.rand([2, 5])

print(x)
print(x.shape)

# Reshape the array
x = x.view([1, 10])
print(x)
