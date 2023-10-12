import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import overall_performance
import numpy as np
from transformers import AutoModel
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

def regulization(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def train(epoch, model, loss_fn, loss_l1, optimizer, train_loader, scheduler=None):
    epoch_start_time = time.time()
    model.train()
    tr_loss = 0  # training loss in current epoch

    # ! training
    for step, batch in enumerate(tqdm(train_loader, desc='Iteration')):
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels = batch
        # compute logits
        out = model(ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels)

        # compute loss
        # loss = loss_fn(out, labels)
        # val_loss += loss.item()
        # print(out)
        # print(F.softmax(out))
        # compute loss
        loss_r = regulization(model, 1e-7)
        loss = loss_fn(out, labels)+loss_r

        tr_loss += loss.item()
        # back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # adjusting learning rate
        optimizer.zero_grad()

    timing = time.time() - epoch_start_time
    cur_lr1 = optimizer.param_groups[0]["lr"]
    cur_lr2 = optimizer.param_groups[1]["lr"]

    print(f"Timing: {timing}, Epoch: {epoch + 1}, training loss: {tr_loss}, current learning rate {cur_lr1}, {cur_lr2}")


def val(model, val_loader):
    # make sure to open the eval mode.
    model.eval()

    # prepare loss function
    #loss_fn = nn.CrossEntropyLoss()

    #val_loss = 0
    val_preds = []
    val_labels = []
    for batch in val_loader:
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels = batch
        # compute logits
        with torch.no_grad():
            # compute logits
            out= model( ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels)
            # get the prediction labels
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()  # prediction labels [1, batch_size]
            # compute loss
            #loss = loss_fn(out, labels)
            #val_loss += loss.item()

            labels = labels.cpu().numpy().tolist()  # ground truth labels [1, batch_size]
            val_labels.extend(labels)
            val_preds.extend(preds)

    # get overall performance
    val_acc, val_prec, val_recall, val_f1 = overall_performance(val_labels, val_preds)
    return val_acc, val_prec, val_recall, val_f1

def exp(model, val_loader):
    # make sure to open the eval mode.
    model.eval()
    x = torch.tensor([])
    y = torch.tensor([])
    for batch in val_loader:
        # unpack batch data

        batch = tuple(t.cuda() for t in batch)
        ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels = batch
        with torch.no_grad():
            out,rou = model( ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels, False)
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
            r = -torch.diagonal(torch.bmm(rou.real.cpu(),torch.log1p(rou.real.cpu())), dim1=1, dim2=2).sum(dim=1)

            y = torch.cat([y, r], dim=0)
            x = torch.cat([x, gloss_att.cpu().sum(-1)], dim=0)
        print('1111')
    print(y)
    print(x)



    min_val = torch.min(y)
    max_val = torch.max(y)
    y = (y - min_val) / (max_val - min_val)

    x = x.detach().numpy()
    y = y.detach().numpy()
    polyfit = np.polyfit(x, y, deg=1)
    p = np.poly1d(polyfit)

    # 计算拟合线附近的阴影区域
    y_fit = p(x)
    y_err = y - y_fit
    mean_err = np.mean(y_err)
    std_err = np.std(y_err)
    interval = 2 * std_err
    y_min = y_fit + mean_err - interval / 2
    y_max = y_fit + mean_err + interval / 2

    # 画出拟合线和阴影区域
    fig, ax = plt.subplots()
    ax.plot(x, y_fit, color='red', label='fit')
    ax.fill_between(x, y_min, y_max, alpha=0.2, label='confidence interval')
    ax.scatter(x, y, label='data', s=25,)
    plt.ylim(0.4, 1)
    plt.xlim(3.5, 9.5)
    plt.xlabel('Number of synsets')
    plt.ylabel('von Neumann entropy')
    plt.savefig('vn.pdf')
    input()


def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_plm(args):
    # loading Pretrained Model
    plm = AutoModel.from_pretrained(args.DATA.plm)
    if args.DATA.use_context:
        config = plm.config
        config.type_vocab_size = 4
        plm.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        plm._init_weights(plm.embeddings.token_type_embeddings)

    return plm