import torch
import torch.nn as nn
import torch.nn.functional as F

class UT(nn.Module):
    def __init__(self, n):
        super(UT, self).__init__()

        self.fc0 = nn.Linear(n, n)
        self.fc1 = nn.Linear(n, n)

    def forward(self, x):
        # Compute the QR decomposition of the matrix
        x = self.fc0(self.fc1(x))
        x = F.normalize(x, dim=-1)
        return x
class UTt(nn.Module):
    def __init__(self, n):
        super(UTt, self).__init__()

        self.models = nn.ModuleList([UT(32) for i in range(1, 21)])

    def forward(self, x, pos):
        # Compute the QR decomposition of the matrix




        tar_u = []
        index = 0
        #context_density = torch.bmm(context.unsqueeze(2), context.unsqueeze(1))
        for p in pos:
            tmp = self.models[p](x[index])
            tar_u.append(tmp)
            index += 1
        tar = torch.stack(tar_u)
        print(tar.shape)

        return tar


# Example usage
n = 4
x = torch.randn(4,32)

x = F.normalize(x, dim=-1)



# Compute the output of the orthogonal matrix
u = UTt(32)
output = u(x,torch.tensor([0,2,1,6]))
print(output)
# Compute the Frobenius norm of the matrix


a = torch.randn(4, 16) # 第一个量子系统
b = torch.randn(4,30, 16) # 第二个量子系统

c = []
for i in range(4):
    c.append(torch.kron(a[i], b[i])) # 计算张量积
c = torch.stack(c)
print(c.shape)