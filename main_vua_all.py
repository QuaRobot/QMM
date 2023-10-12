import os
import os.path as osp
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn

from CNM import QLModel
from data_loader import dataset, collate_fn
from data_process import VUA_All_Processor, Verb_Processor
from utils import Logger
from configs.default import get_config
from train_val import train, val, load_plm, set_random_seeds


def parse_option():
    parser = argparse.ArgumentParser(description='Train on VUA All dataset')
    parser.add_argument('--cfg', type=str, default='./configs/vua_all.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='vua_all', type=str)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output, f'{args.log}.txt'))
    print(args)
    set_random_seeds(args.seed)

    # prepare train-val datasets and dataloaders
    processor = VUA_All_Processor(args)

    # load model
    plm = load_plm(args)
    model = QLModel(args=args, plm=plm, k1=0, k2=0)
    model.cuda()

    if args.eval_mode:
        print("Evaluate only")
        vp = Verb_Processor(args)
        model.load_state_dict(torch.load('./checkpoints/best_vua_all.pth'))
        test_data = processor.get_test_data()
        acad_data = processor.get_acad()
        conv_data = processor.get_conv()
        fict_data = processor.get_fict()
        news_data = processor.get_news()
        adj_data = processor.get_adj()
        adv_data = processor.get_adv()
        noun_data = processor.get_noun()
        verb_data = processor.get_verb()

        test_set = dataset(test_data)
        acad_set = dataset(acad_data)
        conv_set = dataset(conv_data)
        fict_set = dataset(fict_data)
        news_set = dataset(news_data)
        adj_set = dataset(adj_data)
        adv_set = dataset(adv_data)
        noun_set = dataset(noun_data)
        verb_set = dataset(verb_data)

        test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        acad_loader = DataLoader(acad_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        conv_loader = DataLoader(conv_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        fict_loader = DataLoader(fict_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        news_loader = DataLoader(news_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        adj_loader = DataLoader(adj_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        adv_loader = DataLoader(adv_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        noun_loader = DataLoader(noun_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
        verb_loader = DataLoader(verb_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)

        # transfer learning part
        trofi_data = vp.get_trofi()
        trofi_set = dataset(trofi_data)
        trofi_loader = DataLoader(trofi_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)

        print('-------test-------')
        val(model, test_loader)
        print('-------acad-------')
        val(model, acad_loader)
        print('-------conv-------')
        val(model, conv_loader)
        print('-------fict-------')
        val(model, fict_loader)
        print('-------news-------')
        val(model, news_loader)
        print('-------adj-------')
        val(model, adj_loader)
        print('-------adv-------')
        val(model, adv_loader)
        print('-------noun-------')
        val(model, noun_loader)
        print('-------verb-------')
        val(model, verb_loader)
        print('-------trofi-------')
        val(model, trofi_loader)
        return

    # training mode
    train_data = processor.get_train_data()
    val_data = processor.get_val_data()
    test_data = processor.get_test_data()

    train_set = dataset(train_data)
    val_set = dataset(val_data)
    test_set = dataset(test_data)

    train_loader = DataLoader(train_set, batch_size=args.TRAIN.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.TRAIN.val_batch_size, shuffle=False,  collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False,  collate_fn=collate_fn)

    # prepare optimizer
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.TRAIN.lr)

    # prepare scheduler
    # num_train_optimization_steps = len(train_loader) * args.TRAIN.train_epochs
    # scheduler = get_linear_schedule_with_warmup(
    #    optimizer,
    #    num_warmup_steps=int(args.TRAIN.warmup_epochs * len(train_loader)),
    #    num_training_steps=num_train_optimization_steps,
    # )
    lr = args.TRAIN.lr
    lm_params = list(map(id, model.reallm.parameters()))
    qlm_params = filter(lambda p: id(p) not in lm_params, model.parameters())
    optimizer = AdamW([
        {'params': qlm_params, 'lr': lr * 1000},
        #{'params': qlm_params, 'lr': lr},
        {'params': model.reallm.parameters()}], lr=lr)
    # prepare loss function
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, args.TRAIN.class_weight]).cuda())
    l1_loss = nn.L1Loss()

    best_f1 = 0
    for epoch in range(args.TRAIN.train_epochs):
        print('===== Start training: epoch {} ====='.format(epoch + 1))
        train(epoch, model, loss_fn,l1_loss, optimizer, train_loader, None)
        a, p, r, f1 = val(model, val_loader)
        val(model, test_loader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), './best_vua_all.pth')


if __name__ == '__main__':
    args = parse_option()
    main(args)