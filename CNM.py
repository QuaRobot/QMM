import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_dropout
from matplotlib import pyplot as plt
import seaborn as sns

def attention_plot(attention, figsize=(5, 5), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)


    hm = sns.heatmap(attention,

                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     xticklabels=False,
                     yticklabels=False

                     )

    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()

def normalize_pi(tensor):
    min_value, _ = tensor.min(dim=0, keepdim=True)
    max_value, _ = tensor.max(dim=0, keepdim=True)
    norm_tensor = (tensor - min_value) / (max_value - min_value) * 2 * torch.tensor([np.pi],device='cuda') - torch.tensor([np.pi],device='cuda')
    return norm_tensor


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
        return w


class QLModel(nn.Module):
    def __init__(self, args: object, plm: object, k1:object,k2:object) -> object:
        super(QLModel, self).__init__()


        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.gp = nn.Embedding(10,32)
        self.cp = nn.Embedding(145,32)
        self.tp = nn.Embedding(15,32)
        self.reallm = plm
        self.decoder = AutoTokenizer.from_pretrained(args.DATA.plm)


        for name, param in self.reallm.named_parameters():

                    param.requires_grad = True

        #self.dropout1 = complex_dropout(57600)

        self.fc = ComplexLinear(57600, 2)
        self.norms = ComplexBatchNorm2d(2)
        self.normm = ComplexBatchNorm2d(2)
        self.norms2 = ComplexBatchNorm2d(32)
        self.normm2 = ComplexBatchNorm2d(32)
        self.pool = AttentionPoolingWeight(768)
        self.atten = AttentionPoolingWeight(768)


        self.realliner = nn.Linear(768, 32)

        self.cnmip = ComplexConv2d(2, 32, 3, 1)
        self.cnspv = ComplexConv2d(2, 32, 3, 1)


    def forward(self, ids_r, att_r, tar_r, ids_rg, att_rg, pos,gloss_att,_,flag=False):

        batch_size = ids_r.shape[0]
        ids_rg = ids_rg.reshape(batch_size * 10, 10)
        att_rg = att_rg.reshape(batch_size * 10, 10)

        out_r = self.reallm(ids_r, attention_mask=att_r,)
        out_rg = self.reallm(ids_rg, attention_mask=att_rg)

        w_c = self.atten(out_r['last_hidden_state'], att_r)
        w_g = self.pool(out_rg['last_hidden_state'], att_rg)
        gloss = torch.sum(w_g * out_rg['last_hidden_state'], dim=1).view(batch_size,10,-1)

        tar_mask_ls = (tar_r == 1).float()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), out_r['last_hidden_state'])
        tar = torch.sum(H_t, dim=1) / torch.sum(tar_mask_ls, dim=1).unsqueeze(1)

        context = self.realliner(out_r['last_hidden_state'])
        gloss = self.realliner(gloss)
        tar = self.realliner(tar)

        context = F.normalize(context, dim=-1)
        gloss = F.normalize(gloss, dim=-1)
        tar = F.normalize(tar, dim=-1)

        g_image = self.gp(torch.arange(0, 10, device='cuda').expand(batch_size, 10))
        c_image = self.cp(tar_r)
        #t_image = self.tp(pos)

        gloss = gloss.type(torch.complex64)
        context = context.type(torch.complex64)
        tar = tar.type(torch.complex64)

        gloss.imag = g_image
        context.imag = c_image
        #tar.imag = t_image

        gloss_weight = torch.bmm(out_rg['pooler_output'].view(batch_size,10,-1),out_r['pooler_output'].unsqueeze(-1),)

        gloss_weight[gloss_att.unsqueeze(-1) == 0] = float('-inf')
        gloss_weight = torch.softmax(gloss_weight,1)

        w1 = torch.diag_embed(gloss_weight.squeeze()).type(torch.complex64)
        w2 = torch.diag_embed(w_c.squeeze()).type(torch.complex64)

        gloss_density = torch.bmm(torch.bmm(torch.transpose(gloss, 1, 2).conj(),w1),gloss)
        context_density = torch.bmm(torch.bmm(torch.transpose(context, 1, 2).conj(),w2), context)
        tar_density = torch.bmm(tar.unsqueeze(2), tar.unsqueeze(1))

        f_spv = torch.bmm(context_density, tar_density)
        f_mip = torch.bmm(gloss_density, tar_density)

        t_spv = torch.abs(context_density-tar_density)
        t_mip = torch.abs(gloss_density-tar_density)

        spv = torch.stack([t_spv, f_spv], dim=1)
        mip = torch.stack([t_mip, f_mip], dim=1)

        spv = self.norms(spv)
        mip = self.normm(mip)
        spv2 = self.cnspv(spv)
        mip2 = self.cnmip(mip)

        spv2 = spv2.view(batch_size,-1)
        mip2 = mip2.view(batch_size,-1)

        final = torch.cat([spv2,mip2], dim=-1)
        #final = self.dropout1(final)

        out = self.fc(final)
        out = out.abs()

        # if flag is True:
        #     ids_rg = ids_rg.reshape(batch_size, 10, 10)
        #
        #     for i in range(16):
        #         print(self.decoder.decode(ids_r[i]))
        #         for j,k in zip(ids_rg[i],gloss_weight[i]):
        #             print(self.decoder.decode(j), k)
        #         print(_[i])
        #         print(w_c[i])
        #
        #         attention_plot((nor(f_spv[i].cpu()).abs()), figure_name=str(i)+'bmmspv.png')
        #         attention_plot((nor(t_spv[i].cpu()).abs()), figure_name=str(i) + 'absspv.png')
        #         attention_plot((nor(f_mip[i].cpu()).abs()), figure_name=str(i)+'bmmmip.png')
        #         attention_plot((nor(t_mip[i].cpu()).abs()), figure_name=str(i) + 'absmip.png')
        # return out, gloss_density
        
        return out


if __name__ == "__main__":
    pass



