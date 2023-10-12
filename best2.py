import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
# from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
# from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_dropout
#5e-6,0.005,200

def normalize_pi(tensor):
    min_value, _ = tensor.min(dim=0, keepdim=True)
    max_value, _ = tensor.max(dim=0, keepdim=True)
    norm_tensor = (tensor - min_value) / (max_value - min_value) * 2 * torch.tensor([np.pi],device='cuda') - torch.tensor([np.pi],device='cuda')
    return norm_tensor




class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(2, 16, 5, 1)
        self.bn = ComplexBatchNorm2d(16)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 296, 1)
        x = self.bn(x)
        x = x.view(batch_size, -1)


        return x


class QLModel(nn.Module):
    def __init__(self, args: object, plm: object, k1:object,k2:object) -> object:
        super(QLModel, self).__init__()



        self.alpha1 = nn.Parameter(torch.abs(torch.tensor(1.0)))
        self.alpha2 = nn.Parameter(torch.abs(torch.tensor(1.0)))

        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes

        self.reallm = plm
        # self.imaglm = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # self.testi = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32",encodings='utf-8')
        # self.testr = AutoTokenizer.from_pretrained(args.DATA.plm,encodings='utf-8')


        for param in self.reallm.parameters():
            param.requires_grad = True
        # for param in self.imaglm.parameters():
        #     param.requires_grad = True

        self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.5)

        self.fc = nn.Linear(14400,2)
        self.norms =nn.BatchNorm2d(2)
        self.normm =nn.BatchNorm2d(2)
        self.norms2 =nn.BatchNorm2d(64)
        self.normm2 =nn.BatchNorm2d(64)
        self.realliner = nn.Linear(768, 32)
        # self.imagliner = nn.Linear(512, 300)

        self.cnmip = nn.Conv2d(2, 32, 3, 2)
        self.cnspv = nn.Conv2d(2, 32, 3, 2)
        self.poolr = nn.AdaptiveMaxPool2d([30,1])
        self.poolc = nn.AdaptiveMaxPool2d([1,30])
        # self.cnmip2 = nn.Conv2d(32, 1, 3, 1)
        # self.cnspv2 = nn.Conv2d(32, 1, 3, 1)

    def forward(self, ids_r, att_r, tar_r, ids_rg, att_rg, ids_i, att_i, tar_i, ids_ig, att_ig):



        batch_size = ids_r.shape[0]
        ids_rg = ids_rg.reshape(batch_size * 10, 30)
        att_rg = att_rg.reshape(batch_size * 10, 30)
        # ids_ig = ids_ig.reshape(batch_size * 10, 30)
        # att_ig = att_ig.reshape(batch_size * 10, 30)


        out_r = self.reallm(ids_r, attention_mask=att_r)
        # out_i = self.imaglm(ids_i, attention_mask=att_i)

        out_rg = self.reallm(ids_rg, attention_mask=att_rg)
        # out_ig = self.imaglm(ids_ig, attention_mask=att_ig)

        context_real = out_r['pooler_output']
        # context_imag = out_i['pooler_output']

        gloss_real = out_rg['pooler_output'].view(batch_size, 10, 768)
        # gloss_imag = out_ig['pooler_output'].view(batch_size, 10, 512)

        tar_mask_ls = (tar_r == 1).float()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), out_r['last_hidden_state'])
        tar_real = torch.sum(H_t, dim=1) / torch.sum(tar_mask_ls, dim=1).unsqueeze(1)
        # tar_mask_ls = (tar_i == 1).float()
        # H_t = torch.mul(tar_mask_ls.unsqueeze(2), out_i['last_hidden_state'])
        # tar_imag = torch.sum(H_t, dim=1) / torch.sum(tar_mask_ls, dim=1).unsqueeze(1)

        context_real = self.realliner(context_real)
        gloss_real = self.realliner(gloss_real)
        tar_real = self.realliner(tar_real)

        # context_imag = self.imagliner(context_imag)
        # gloss_imag = self.imagliner(gloss_imag)
        # tar_imag = self.imagliner(tar_imag)

        context = F.normalize(context_real, dim=-1)
        gloss = F.normalize(gloss_real, dim=-1)
        tar = F.normalize(tar_real, dim=-1)

        # context.imag = context_imag
        # gloss.imag = gloss_imag
        # tar.imag = tar_imag

        # context_imag = normalize_pi(context_imag)
        # gloss_imag = normalize_pi(gloss_imag)
        # tar_imag = normalize_pi(tar_imag)

        gloss_weight = torch.arange(1, 11, device='cuda').repeat(batch_size).reshape(batch_size, 10)
        gloss_weight = ((torch.exp(-(gloss_weight)*self.alpha2))) \
                       / torch.sum((torch.exp(-(gloss_weight)*self.alpha2)), dim=1).unsqueeze(1)
        P = torch.diag_embed(gloss_weight)
        gloss_density = torch.bmm(torch.bmm(torch.transpose(gloss,1 ,2 ),P ), gloss)

        context_density = torch.bmm(context.unsqueeze(2), context.unsqueeze(1))

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

        # mipr = self.poolr(mip)
        # mipc = self.poolc(mip)
        # spvr = self.poolr(spv)
        # spvc = self.poolc(spv)
        #print(mip.shape)
        # spv = self.norms2(spv)
        # mip = self.normm2(mip)
        # spv = self.cnspv2(spv)
        # mip = self.cnmip2(mip)
        spv = spv.view(batch_size,-1)
        mip = mip.view(batch_size,-1)
        # spvc = spvc.view(batch_size,-1)
        # mipc = mipc.view(batch_size,-1)

        final = torch.cat([spv,mip], dim=-1)
        out = self.dropout1(final)
        out = self.fc(out)

        # out = out.abs()
        # out = F.log_softmax(out, dim=1)
        return out



if __name__ == "__main__":
    pass



