import time
import math
import copy
import wandb
import argparse
import datetime
import random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import json5
from easydict import EasyDict
import os
import gc

from ema_pytorch import EMA
from zinc_model import SpecformerZINC
from large_model import SpecformerLarge
from medium_model import SpecformerMedium
from small_model import SpecformerSmall
from get_dataset import DynamicBatchSampler, RandomSampler, collate_pad, collate_dgl, get_dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_config_from_json(json_file):
    with open('config/' + json_file + '.json', 'r') as config_file:
        config_dict = json5.load(config_file)
    config = EasyDict(config_dict)

    return config
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(dataset, model, device, dataloader, loss_fn, optimizer, wandb=None, wandb_item=None):
    model.train()

    for i, data in enumerate(dataloader):
        e, u, g, length, y = data
        e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)
        # y = y.reshape(-1, 1)

        logits = model(e, u, g, length)
        optimizer.zero_grad()

        # y_idx = y == y # ?
        loss = loss_fn(logits.to(torch.float32), F.one_hot(y, model.nclass).to(torch.float32))

        loss.backward()
        optimizer.step()

        if wandb:
            wandb.log({wandb_item: loss.item()})

def evaluate(scores, targets):
    predictions = scores.argmax(dim=1)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="micro")
    f1 = f1_score(targets, predictions, average="micro")
    macro_f1 = f1_score(targets, predictions, average="macro")
    probs = F.softmax(scores, dim=1)

    unique_values = np.unique(targets)
    if len(unique_values) == scores.shape[1]:
        if scores.shape[1] == 2:
            probs = probs[:, 1]
        roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    else:
        roc = 0

    
    # if scores.shape[1] == 2:
    #     probs = probs[:, 1]
    # roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro f1": macro_f1,
        "roc": roc,
    }

def eval_epoch(dataset, model, device, dataloader, evaluator, metric):
    model.eval()

    y_true = []
    y_pred = []
    scores = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            e, u, g, length, y = data
            e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)

            logits = model(e, u, g, length)
            if model.nclass > 1:
                score = logits.detach()
                prediction = score.argmax(dim=1)
            else:
                score = prediction = logits.detach()

            y_true.append(y.view(prediction.shape).detach().cpu())
            y_pred.append(prediction.cpu())
            scores.append(score.cpu())
        
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    scores = torch.cat(scores, dim=0)
    
    full_eval = evaluate(scores, y_true)

    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred}), full_eval


def main_worker(args, datainfo=None):
    rank = 'cpu' if args.cuda == -1 else 'cuda:{}'.format(args.cuda)
    print(args)

    if datainfo is None:
        datainfo = get_dataset(args.dataset)
    nclass = datainfo['num_class']
    loss_fn = datainfo['loss_fn']
    evaluator = datainfo['evaluator']
    train = datainfo['train_dataset']
    valid = datainfo['valid_dataset']
    test  = datainfo['test_dataset']
    metric = datainfo['metric']
    metric_mode = datainfo['metric_mode']
    num_node_labels = datainfo.get('num_node_labels')
    num_edge_labels = datainfo.get('num_edge_labels')

    # dataloader
    '''
    train_batch_sampler = DynamicBatchSampler(RandomSampler(train), [data.num_nodes for data in train],
                                              batch_size=32, max_nodes=50, drop_last=False)
    valid_batch_sampler = DynamicBatchSampler(RandomSampler(valid), [data.num_nodes for data in valid],
                                              batch_size=32, max_nodes=50, drop_last=False)
    test_batch_sampler  = DynamicBatchSampler(RandomSampler(test),  [data.num_nodes for data in test],
                                              batch_size=32, max_nodes=50, drop_last=False)
    train_dataloader = DataLoader(train, batch_sampler=train_batch_sampler, collate_fn=collate_pad)
    valid_dataloader = DataLoader(valid, batch_sampler=valid_batch_sampler, collate_fn=collate_pad)
    test_dataloader  = DataLoader(test,  batch_sampler=test_batch_sampler,  collate_fn=collate_pad)
    '''

    half_batch_size = max(args.batch_size // 2, 1)
    train_dataloader = DataLoader(train, batch_size = args.batch_size, num_workers=4, collate_fn=collate_dgl, shuffle = True)
    valid_dataloader = DataLoader(valid, batch_size = half_batch_size, num_workers=4, collate_fn=collate_dgl, shuffle = False)
    test_dataloader  = DataLoader(test,  batch_size = half_batch_size, num_workers=4, collate_fn=collate_dgl, shuffle = False)
    
    # print("Data loader ready", torch.cuda.memory_allocated())
    # del train, datainfo['train_dataset']
    # del valid, datainfo['valid_dataset']
    # del test, datainfo['test_dataset']
    torch.cuda.empty_cache()
    gc.collect()

    # print("Data loader clean", torch.cuda.memory_allocated())

    if args.dataset == 'zinc':
        print('zinc')
        model = SpecformerZINC(nclass, args.nlayer, args.hidden_dim, args.nheads,
                               args.feat_dropout, args.trans_dropout, args.adj_dropout)

    elif args.dataset == 'pcqm' or args.dataset == 'pcqms':
        print('pcqm')
        model = SpecformerLarge(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                 args.feat_dropout, args.trans_dropout, args.adj_dropout)
        print('init')
        model.apply(init_params)

    elif args.dataset == 'pcba':
        print('pcba')
        model = SpecformerMedium(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                 args.feat_dropout, args.trans_dropout, args.adj_dropout)
        model.apply(init_params)

    elif args.dataset == 'hiv':
        print('hiv')
        model = SpecformerSmall(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                args.feat_dropout, args.trans_dropout, args.adj_dropout)
    else:
        print(f'TUDataset - {args.dataset}')
        if args.model == "small":
            model = SpecformerSmall(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                    args.feat_dropout, args.trans_dropout, args.adj_dropout, 
                                    num_node_labels, num_edge_labels)
        elif args.model == "medium":
            model = SpecformerMedium(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                    args.feat_dropout, args.trans_dropout, args.adj_dropout,
                                    num_node_labels, num_edge_labels)
        elif args.model == "large":
            model = SpecformerLarge(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                    args.feat_dropout, args.trans_dropout, args.adj_dropout,
                                    num_node_labels, num_edge_labels)
        else:
            raise NotImplementedError()
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(rank)

    # print("Model ready", torch.cuda.memory_allocated())

    print(count_parameters(model))
    
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # warm_up + cosine weight decay
    lr_plan = lambda cur_epoch: (cur_epoch+1) / args.warm_up_epoch if cur_epoch < args.warm_up_epoch else \
              (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_epoch) / (args.epochs - args.warm_up_epoch))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    results = []
    for epoch in range(args.epochs):

        # print("Epoch", epoch, torch.cuda.memory_allocated())

        train_epoch(args.dataset, model, rank, train_dataloader, loss_fn, optimizer, wandb=None, wandb_item='loss')
        scheduler.step()

        # torch.save(model.state_dict(), 'checkpoint/{}_{}.pth'.format(args.project_name, epoch))

        if epoch % args.log_step == 0:

            val_eval, _ = eval_epoch(args.dataset, model, rank, valid_dataloader, evaluator, metric)
            test_eval, full_eval = eval_epoch(args.dataset, model, rank, test_dataloader, evaluator, metric)
            val_res = val_eval[metric]
            test_res = test_eval[metric]

            results.append([val_res, test_res, full_eval])

            if metric_mode == 'min':
                best = sorted(results, key = lambda x: x[0], reverse=False)[0]
                best_res = best[1]
                best_eval = best[2]
            else:
                best = sorted(results, key = lambda x: x[0], reverse=True)[0]
                best_res = best[1]
                best_eval = best[2]

            print(epoch, 
                  'valid: {:.4f}'.format(val_res), 'test: {:.4f}'.format(test_res), 'best: {:.4f}'.format(best_res), 
                  'valid_f1: {:.4f}'.format(val_eval["f1"]), 'test_f1: {:.4f}'.format(test_eval["f1"]), 'best_f1: {:.4f}'.format(best_eval["macro f1"]))

            # wandb.log({'val': val_res, 'test': test_res})
        
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), 'checkpoint/{}.pth'.format(args.project_name))

    return model, rank


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--cuda', type=int, default=0)
#     parser.add_argument('--dataset', default='zinc')
#     parser.add_argument('--model', default='small')

#     args = parser.parse_args()
#     args.project_name = datetime.datetime.now().strftime('%m-%d-%X')

#     # config = get_config_from_json(args.dataset)

#     config = {
#         "nlayer": 4,
#         "nheads": 8,
#         "hidden_dim": 160,
#         "trans_dropout": 0.1,
#         "feat_dropout": 0.05,
#         "adj_dropout": 0.0,
#         "lr": 1e-3,
#         "weight_decay": 5e-4,
#         "epochs": 1,
#         "warm_up_epoch": 50,
#         "batch_size": 32
#     }


#     for key in config.keys():
#         setattr(args, key, config[key])

#     main_worker(args)

