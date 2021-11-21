"""
Example usage for doc-level detection task:
python code/engine.py --task doc-level --data_dir [data/NYTimes/] --lrate 5e-6 --num_epochs 4
python code/engine.py --task doc-level --data_dir data/VOA/ --lrate 5e-6 --num_epochs 4 --batch_size 2

Example usage for KE-level detection task:
python code/engine.py --task KE-level --data_dir data/VOA/ --lrate 0.001
"""
import os
import re
import dgl
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from model import InfoSurgeon
from torch.utils.data import DataLoader
from KE_dataset_new import KEDataset
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--task', nargs="?", type=str, default="doc-level")
parser.add_argument('--device', nargs="?", type=str, default="cpu")
parser.add_argument('--batch_size', nargs="?", type=int, default=1)
parser.add_argument('--mode', nargs="?", type=str, default="train")
parser.add_argument('--lrate', type=float, default=2e-5)
parser.add_argument('--num_epochs', nargs="?", type=int, default=1)
parser.add_argument('--load_model_checkpoint', nargs="?", type=bool, default=False)
parser.add_argument('--ckpt_name', nargs="?", type=str, default="NYTimes")
parser.add_argument('--use_IE', nargs="?", type=bool, default=True)
parser.add_argument('--data_dir', nargs="*", type=str, default=["data/NYTimes/"])
args = parser.parse_args()

device, batch_size = args.device, args.batch_size
mode, num_epochs = args.mode, args.num_epochs
load_model_checkpoint, ckpt_name = args.load_model_checkpoint, args.ckpt_name
use_IE, use_gen_f = args.use_IE, True
DATA_DIR = args.data_dir

with open('NLP_toolbox/glove.840B.300d.pickle', 'rb') as f:
    glove_embed = pickle.load(f)

def collate(samples):
    try:
        samples = list(filter (lambda x: x is not None, samples))
        src, tgt, segs, clss, im_data, cap_data, title_data, ind_facs, KE_labels, labels, articleIDs, g = map(list, zip(*samples)) 
        g = [x.to(device) for x in g]
        output1, output2, output3 = (g,), (), ()
        if use_gen_f:
            combined_caps = []
            for i in cap_data:
                 combined_caps += i
            cap_src, cap_tgt = [x[0] for x in combined_caps], [x[1] for x in combined_caps]
            cap_segs, cap_clss = [x[2] for x in combined_caps], [x[3] for x in combined_caps]
            title_src, title_tgt = [x[0] for x in title_data], [x[1] for x in title_data] 
            title_segs, title_clss = [x[2] for x in title_data], [x[3] for x in title_data]
            src, cap_src  = torch.tensor(_pad(src, 0)), torch.tensor(_pad(cap_src, 0))
            tgt, cap_tgt = torch.tensor(_pad(tgt, 0)), torch.tensor(_pad(cap_tgt, 0))
            title_src, title_tgt = torch.tensor(_pad(title_src, 0)), torch.tensor(_pad(title_tgt, 0))
            segs, cap_segs = torch.tensor(_pad(segs, 0)), torch.tensor(_pad(cap_segs, 0))
            title_segs = torch.tensor(_pad(title_segs, 0))
            mask_src, mask_tgt, cap_mask_src, cap_mask_tgt = ~(src == 0), ~(tgt == 0), ~(cap_src == 0), ~(cap_tgt == 0)
            title_mask_src, title_mask_tgt = ~(title_src == 0), ~(title_tgt == 0)
            clss, cap_clss = torch.tensor(_pad(clss, -1)), torch.tensor(_pad(cap_clss, -1))
            mask_cls, cap_mask_cls = ~(clss == -1), ~(cap_clss == -1)
            title_clss, title_mask_cls = torch.tensor(_pad(title_clss, -1)), torch.tensor(~(title_clss == -1))
            clss[clss == -1], cap_clss[cap_clss == -1], title_clss[title_clss == -1] = 0, 0, 0
            main_text = (src,tgt,segs,clss,mask_src,mask_tgt,mask_cls)
            im_data = torch.tensor(im_data).float()
            cap_text = (cap_src,cap_tgt,cap_segs,cap_clss,cap_mask_src,cap_mask_tgt,cap_mask_cls)
            title_text = (title_src,title_tgt,title_segs,title_clss,title_mask_src,title_mask_tgt,title_mask_cls)
            ind_facs = torch.tensor(ind_facs).float()
            output2 = (main_text, im_data, cap_text, title_text, ind_facs)
        output3 = (torch.tensor(labels).float().to(device), articleIDs)
        return output1 + output2 + output3
    except:
        return None

def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    return [d + [pad_id] * (width - len(d)) for d in data]

data = (glove_embed, DATA_DIR)
if mode == "train":
    KE_dataset_val = KEDataset(data, "val", use_IE, use_gen_f)
    dataloader_val = DataLoader(KE_dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    KE_dataset_train = KEDataset(data, "train", use_IE, use_gen_f)
    dataloader_train = DataLoader(KE_dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
KE_dataset_test = KEDataset(data, "test", use_IE, use_gen_f)
dataloader_test = DataLoader(KE_dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

sig = nn.Sigmoid()
model = InfoSurgeon(use_IE, use_gen_f, device).to(device)
if load_model_checkpoint:
    model.load_state_dict(torch.load("checkpoint/KE_detector_"+ckpt_name+".pt"))
    print("Loaded model")
if args.task == "KE-level":
    model.loss = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=args.lrate)

for epoch in range(num_epochs):
    model.train()
    acc, tot_count = 0.0, 0.001
    rel_sel_count, sel_count, rel_count, TN = 0.0, 0.0, 0.0, 0.0
    for idx, stuff in enumerate(dataloader_train):
        if stuff is None:
            continue
        if True: #try:
            optim.zero_grad()
            g, bert_data, im_data, cap_data, title_data, ind_facs, labels, articleIDs = stuff
            logits, G = model((g, bert_data, im_data, cap_data, title_data, ind_facs))
            if args.task == "KE-level":
                logits = G.edges['local_edge'].data['score'][((G.edges['local_edge'].data["local_train_flag"]==1) & (G.edges['local_edge'].data["doc_label"]==1))]
                labels = G.edges['local_edge'].data['labels'].float()[((G.edges['local_edge'].data["local_train_flag"]== 1) & (G.edges['local_edge'].data["doc_label"]==1))]
            if args.task == "KE-level" and logits.size()[0] == 0:
                continue
            loss = model.compute_loss(logits, labels)
            loss.backward()
            optim.step()
            pred_labels = logits.argmax(dim=1)
            acc += (pred_labels==labels).sum()
            tot_count += pred_labels.size()[0]
            if args.task == "KE-level":
                rel_sel_count += ((pred_labels == 1) & (labels == 1)).sum()
                sel_count += (pred_labels == 1).sum()
                rel_count += (labels == 1).sum()
                TN += ((pred_labels == 0) & (labels == 0)).sum()
            if idx % 10 == 0:
                print(epoch, idx, acc/tot_count)
                if args.task == "KE-level":
                    print(rel_sel_count, sel_count, rel_count, TN)
                torch.save(model.state_dict(), "checkpoint/KE_detector_"+ckpt_name+".pt")
        #except:
        #    print("Failed in main...")
    print("Epoch, Train Accuracy, Total Count: ", epoch, acc/tot_count, tot_count)
    model.eval()
    acc, tot_count = 0.0, 0.001
    rel_sel_count, sel_count, rel_count, TN = 0.0, 0.0, 0.0, 0.0
    for idx, stuff in enumerate(dataloader_val):
        if stuff is None:
            continue
        if True: #try:
            g, bert_data, im_data, cap_data, title_data, ind_facs, labels, articleIDs = stuff
            logits, G = model((g, bert_data, im_data, cap_data, title_data, ind_facs))
            if args.task == "KE-level":
                logits = G.edges['local_edge'].data['score'][G.edges['local_edge'].data["doc_label"]== 1]
                labels = G.edges['local_edge'].data['labels'].float()[G.edges['local_edge'].data["doc_label"]== 1]
            if args.task == "KE-level" and logits.size()[0] == 0:
                continue
            pred_labels = logits.argmax(dim=1)
            acc += (pred_labels==labels).sum()
            tot_count += labels.size()[0]
            if args.task == "KE-level":
                print(pred_labels, labels)
                print(rel_sel_count, sel_count, rel_count)
                rel_sel_count += ((pred_labels == 1) & (labels == 1)).sum()
                sel_count += (pred_labels == 1).sum()
                rel_count += (labels == 1).sum()
                TN += ((pred_labels == 0) & (labels == 0)).sum()
        #except:
        #    pass
        print(acc/tot_count, acc, tot_count)
    print("Val Accuracy: ", acc/tot_count, acc, tot_count)
    if args.task == "KE-level":
        p, r = rel_sel_count/(sel_count+0.001), rel_sel_count/(rel_count+0.001)
        print(epoch, "P, R, Fscore: ", p, r, 2*(p*r)/(p+r+0.0001))
results_out = {}
for idx, stuff in enumerate(dataloader_test):
    if stuff is None:
        continue
    g, bert_data, im_data, cap_data, title_data, ind_facs, labels, articleIDs = stuff
    logits, G = model((g, bert_data, im_data, cap_data, title_data, ind_facs))
    if args.task == "KE-level":
        logits = G.edges['local_edge'].data['score'][G.edges['local_edge'].data["local_train_flag"]== 1]
        labels = G.edges['local_edge'].data['labels'].float()[G.edges['local_edge'].data["local_train_flag"]== 1]
    pred_labels = logits.argmax(dim=1)
    acc += (pred_labels==labels).sum()
    tot_count += label.size()[0]
    if args.task == "KE-level":
        rel_sel_count += ((pred_labels == 1) & (labels == 1)).sum()
        sel_count += (pred_labels == 1).sum()
        rel_count += (labels == 1).sum()
        TN += ((pred_labels == 0) & (labels == 0)).sum()
print("Test Accuracy: ", acc/tot_count, acc, tot_count)
if args.task == "KE-level":
    p, r = rel_sel_count/(sel_count+0.001), rel_sel_count/(rel_count+0.001)
    print(epoch, "P, R, Fscore: ", p, r, 2*(p*r)/(p+r+0.0001))
