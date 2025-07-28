# import torch
# import torch.nn as nn

# linear = nn.Linear(768, 768).cuda()
# inp = torch.randn(10, 768, device='cuda')
# out = linear(inp)
# print(out.shape)
# assert out.shape == (10, 768), "Output shape mismatch: expected (10, 768), got {}".format(out.shape)


# import torch
# import torch.nn as nn

# # force inputs on CPU
# input_ids = torch.randint(0, 10000, (32, 64))

# # embedding on GPU
# embedding = nn.Embedding(10000, 768).cuda()

# # move input to GPU
# input_ids = input_ids.cuda()

# # run
# output = embedding(input_ids)
# print(output.shape)


# import torch

# # This should NOT trigger cuRAND
# inp = torch.empty(10, 768, device='cuda')
# print(inp.shape)


# import torch

# # This WILL trigger cuRAND
# inp = torch.randn(10, 768, device='cuda')
# print(inp.shape)




# import torch

# x = torch.cuda.FloatTensor(1000)  # Only allocate, no .uniform_()

# print("Allocated tensor:", x)


# import torch

# x = torch.cuda.FloatTensor(10).uniform_()

# print("Generated tensor:", x)


# finally this works!
# import torch

# x = torch.tensor(10, dtype=torch.float32, device='cuda')

# print("Generated tensor:", x)


# THIS NEEDS TO BE FIXED
# import torch
# import torch.nn as nn

# m = nn.Linear(4, 4).cuda()
# x = torch.rand(1, 4, device="cuda")

# print("About to run linear layer...")
# y = m(x)
# print("Output:", y)

import torch
a = torch.randn(4, 4, device="cuda")
b = torch.randn(4, 4, device="cuda")
c = torch.matmul(a, b)

print("Matrix multiplication result:", c)

# import torch
# print(torch.cuda.memory_allocated())
# torch.empty(100, device='cuda')
# print(torch.cuda.memory_allocated())
