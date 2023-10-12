import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#5e-6,50





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



        self.alpha2 = nn.Parameter(torch.abs(torch.tensor(1.0)))

        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes

        self.reallm = plm


        for param in self.reallm.parameters():
            param.requires_grad = True


        self.dropout1 = nn.Dropout(0.5)


        self.fc = nn.Linear(288,2)
        self.norms = nn.BatchNorm2d(2)
        self.normm = nn.BatchNorm2d(2)

        self.pool = AttentionPoolingWeight(768)
        self.atten = AttentionPoolingWeight(768)


        self.realliner = nn.Linear(768,32)
        # self.imagliner = nn.Linear(512, 300)

        self.cnmip = nn.Conv2d(2, 64, 3, 1)
        self.cnspv = nn.Conv2d(2, 64, 3, 1)


    def forward(self, ids_r, att_r, tar_r, ids_rg, att_rg, pos,_):

        batch_size = ids_r.shape[0]
        ids_rg = ids_rg.reshape(batch_size * 10, 30)
        att_rg = att_rg.reshape(batch_size * 10, 30)

        out_r = self.reallm(ids_r, attention_mask=att_r)
        out_rg = self.reallm(ids_rg, attention_mask=att_rg)


        out_r = out_r.last_hidden_state
        out_rg = out_rg.last_hidden_state


        w_c = self.atten(out_r, att_r)
        w_g = self.pool(out_rg, att_rg)
        gloss = torch.sum(w_g * out_rg, dim=1).view(batch_size,10,-1)
        #context = torch.sum(w_c * out_r['last_hidden_state'], dim=1)


        tar_mask_ls = (tar_r == 1).float()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), out_r)
        tar = torch.sum(H_t, dim=1) / torch.sum(tar_mask_ls, dim=1).unsqueeze(1)

        context = self.realliner(out_r)
        gloss = self.realliner(gloss)
        tar = self.realliner(tar)

        context = F.normalize(context, dim=-1)
        gloss = F.normalize(gloss, dim=-1)
        tar = F.normalize(tar, dim=-1)

        #tar = self.pos(tar, pos)

        gloss_weight = torch.arange(1, 11, device='cuda').repeat(batch_size).reshape(batch_size, 10)
        gloss_weight = ((torch.exp(-(gloss_weight)*self.alpha2))) \
                       / torch.sum((torch.exp(-(gloss_weight)*self.alpha2)), dim=1).unsqueeze(1)

        P = torch.diag_embed(gloss_weight)
        gloss_density = torch.bmm(torch.bmm(torch.transpose(gloss,1 ,2 ),P ), gloss)
        P = torch.diag_embed(w_c.squeeze())
        context_density = torch.bmm(torch.bmm(torch.transpose(context,1 ,2 ),P ), context)


        #context_density = torch.bmm(context.unsqueeze(2), context.unsqueeze(1))

        tar_density = torch.bmm(tar.unsqueeze(2), tar.unsqueeze(1))


        f_spv = torch.bmm(context_density, tar_density)
        f_mip = torch.bmm(gloss_density, tar_density)

        t_spv = torch.abs(context_density-tar_density)
        t_mip = torch.abs(gloss_density-tar_density)

        spv = torch.stack([t_spv, f_spv], dim=1)
        mip = torch.stack([t_mip, f_mip], dim=1)
        spv = self.norms(spv)
        mip = self.normm(mip)
        spv = self.cnspv(spv)
        mip = self.cnmip(mip)

        spv = spv.view(batch_size,-1)
        mip = mip.view(batch_size,-1)

        final = torch.cat([spv,mip], dim=-1)
        out = self.dropout1(final)
        out = self.fc(out)

        return out



if __name__ == "__main__":
    pass



