import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Measure(nn.Module):
    def __init__(self, k,n):
        super(Measure, self).__init__()
        self.n = n
        self.k = k
        self.M = nn.Parameter(torch.rand(k,n,n))

    def forward(self, x):
        # Compute the QR decomposition of the matrix

        res = []
        for i in range(self.k):
            m = torch.mul(torch.transpose(self.M[i], 0, 1), self.M[i])
            x = torch.bmm(m.repeat(16), x)
            res.append(torch.einsum('bii->b',x).unsqueeze(-1).unsqueeze(-1))
        return torch.stack(res)

class MeasureNet(nn.Module):  # 继承父类，nn.Module类
    def __init__(self, times, d, poststate=False):  # 初始化函数
        """
            input:times测量次数，d状态向量维度,poststate是否返回测量后的密度矩阵，默认不反回
            output：测量结果，测量后的密度矩阵

        """
        super(MeasureNet, self).__init__()  # 继承nn.Model类的初始化函数
        self.org = nn.Parameter(torch.FloatTensor(times, d), requires_grad=True)  # 初始化投影算子
        self.org.data.normal_(mean=0, std=0.1)
        self.post = poststate

    def forward(self, x):
        org = F.normalize(self.org, dim=-1)  # 保证酉性质
        m = torch.einsum('ab,ac->abc', [org, org])
        x = torch.einsum('abc,dce->dabe', [m, x])
        trace = torch.einsum('abcc->ab', [x])

        return trace

class EMB(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.W = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, in_dim),
        )
    def forward(self, x):
        x = self.W(x)
        x = F.normalize(x, dim=-1)
        return x


class AttentionPoolingWeight(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )
    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        return torch.sum(w * last_hidden_state, dim=1)

class AttentionWeight(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )
    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)

        return w


class QLModel(nn.Module):
    def __init__(self, args: object, plm: object, k1:object,k2:object) -> object:
        super(QLModel, self).__init__()

        self.alpha2 = nn.Parameter(torch.abs(torch.tensor(1.0)))

        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes

        self.reallm = plm

        for param in self.reallm.parameters():
            param.requires_grad = True

        self.MS = MeasureNet(300, 256)
        self.MM = MeasureNet(300, 256)

        self.pool = AttentionPoolingWeight(768)
        self.atten = AttentionWeight(768)
        self.ee1 = EMB(256)
        self.ee2 = EMB(256)
        self.fc = nn.Linear(600,2)

        self.realliner = nn.Linear(768, 16)


    def forward(self, ids_c, att_c, tar, ids_g, att_g, pos, _):

        batch_size = ids_c.shape[0]
        ids_g = ids_g.reshape(batch_size * 10, 10)
        att_g = att_g.reshape(batch_size * 10, 10)

        out_c = self.reallm(ids_c, attention_mask=att_c)
        out_g = self.reallm(ids_g, attention_mask=att_g)


        context = out_c['last_hidden_state']

        gloss = self.pool(out_g['last_hidden_state'], att_g).view(batch_size,10,-1)

        tar_mask_ls = (tar == 1).float()
        con_atten = (tar <= 12).float()

        H_t = torch.mul(tar_mask_ls.unsqueeze(2), out_c['last_hidden_state'])
        tar = torch.sum(H_t, dim=1) / torch.sum(tar_mask_ls, dim=1).unsqueeze(1)

        context = self.realliner(context)
        gloss = self.realliner(gloss)
        tar = self.realliner(tar)

        context = F.normalize(context, dim=-1)
        gloss = F.normalize(gloss, dim=-1)
        tar = F.normalize(tar, dim=-1)

        c = []
        g = []
        for i in range(batch_size):
            c.append(torch.kron(tar[i], context[i]))  # 计算张量积
            g.append(torch.kron(tar[i], gloss[i]))  # 计算张量积

        spv = torch.stack(c)
        mip = torch.stack(g)

        spv = self.ee1(spv)
        mip = self.ee2(mip)

        gloss_weight = torch.arange(1, 11, device='cuda').repeat(batch_size).reshape(batch_size, 10)
        gloss_weight = ((torch.exp(-(gloss_weight)*self.alpha2))) \
                       / torch.sum((torch.exp(-(gloss_weight)*self.alpha2)), dim=1).unsqueeze(1)


        w_c = self.atten(out_c['last_hidden_state'], con_atten-tar_mask_ls)


        w1 = torch.diag_embed(gloss_weight)
        mip_density = torch.bmm(torch.bmm(torch.transpose(mip,1, 2 ),w1), mip)
        w2 = torch.diag_embed(w_c.squeeze())
        spv_density = torch.bmm(torch.bmm(torch.transpose(spv,1, 2 ),w2), spv)

        mip = self.MM(mip_density)
        spv = self.MS(spv_density)

        final = torch.cat([spv,mip], dim=-1)
        out = self.fc(final)

        return out


if __name__ == "__main__":
    pass



