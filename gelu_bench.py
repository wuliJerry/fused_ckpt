import torch
import time
import numpy as np

def GeLu(x):
    return 0.5 * x * \
    (1 + torch.tanh(0.797885 * (x + 0.044715 * torch.pow(x, 3))))

# generating the tensor input with 2^25 data
x = torch.randn(2**25, requires_grad=True)

# do the forward pass and backward pass without checkpoints
start = time.time()
y = GeLu(x)
z = y.sum()
z.backward()
end = time.time()

print("Time taken without checkpointing: ", end - start)

# do the forward pass and backward pass with checkpoints
start = time.time()
y = GeLu(x)
y = y.detach()
y.requires_grad_()
z = y.sum()
z.backward()
end = time.time()

print("Time taken with checkpointing: ", end - start)
