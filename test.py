import numpy as np

import torch

# def normalize_tensor(tensor):
#     min_value, _ = tensor.min(dim=0, keepdim=True)
#     max_value, _ = tensor.max(dim=0, keepdim=True)
#     print(max_value)
#     norm_tensor = (tensor - min_value) / (max_value - min_value) * 2 * torch.tensor([np.pi]) - torch.tensor([np.pi])
#     return norm_tensor
#
# # 创建一个随机张量
# tensor = torch.rand((4, 3))
#
# # 归一化张量
# norm_tensor = normalize_tensor(tensor)
#
# # 打印原始张量和归一化后的张量
# print("原始张量:\n", tensor)
# print("归一化张量:\n", norm_tensor)

b = torch.arange(10).type(torch.complex64)
b.imag = torch.arange(10)
print(b)