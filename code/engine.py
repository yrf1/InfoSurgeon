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
parser.add_argument('device', nargs="?", type=str, default="cpu")
parser.add_argument('batch_size', nargs="?", type=int, default=1)
parser.add_argument('mode', nargs="?", type=str, default="train")
parser.add_argument('num_epochs', nargs="?", type=int, default=4)
parser.add_argument('load_model_checkpoint', nargs="?", type=bool, default=False)
parser.add_argument('ckpt_name', nargs="?", type=str, default="NYTimes")
parser.add_argument('use_IE', nargs="?", type=bool, default=False)
parser.add_argument('data_dir', nargs="?", type=list, default=["data/NYTimes/"])
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
        if use_IE and use_gen_f:
            graphs, src, tgt, segs, clss, im_data, cap_data, title_data, ind_facs, KE_labels, labels, articleIDs = map(list, zip(*samples))
        elif use_IE == False:
            src, tgt, segs, clss, im_data, cap_data, title_data, ind_facs, KE_labels, labels, articleIDs, g = map(list, zip(*samples))
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
            title_clss, title_mask_cls = torch.tensor(_pad(title_clss, -1)), ~(title_clss == -1)
            clss[clss == -1], cap_clss[cap_clss == -1], title_clss[title_clss == -1] = 0, 0, 0
            main_text = (src,tgt,segs,clss,mask_src,mask_tgt,mask_cls)
            im_data = torch.tensor(im_data).float()
            cap_text = (cap_src,cap_tgt,cap_segs,cap_clss,cap_mask_src,cap_mask_tgt,cap_mask_cls)
            title_text = (title_src,title_tgt,title_segs,title_clss,title_mask_src,title_mask_tgt,title_mask_cls)
            ind_facs = torch.tensor(ind_facs).float()
            output2 = (main_text, im_data, cap_text, title_text, ind_facs)
        output3 = (torch.tensor(KE_labels).float(), torch.tensor(labels).float(), articleIDs)
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
optim = torch.optim.Adam(model.parameters(), lr=5e-6)

for epoch in range(num_epochs):
    if mode == "train":
        acc, tot_count = 0.0, 0.001
        for idx, stuff in enumerate(dataloader_train):
            if stuff is None:
                continue
            g, bert_data, im_data, cap_data, title_data, ind_facs, KE_label, label, articleIDs = stuff
            pred_y_doc = model((g, bert_data, im_data, cap_data, title_data, ind_facs))
            optim.zero_grad()
            loss = model.compute_loss(pred_y_doc, label.to(device))
            loss.backward()
            optim.step()
            pred_y_doc = torch.argmax(pred_y_doc, dim=-1)
            pred_y_doc = pred_y_doc.to("cpu")
            acc += (pred_y_doc.eq(label)).sum()
            tot_count += label.size()[0]
            if idx % 200 == 0:
                print(idx, pred_y_doc, label, acc/tot_count)
                torch.save(model.state_dict(), "checkpoint/KE_detector_"+ckpt_name+".pt")
        print("Epoch, Train Accuracy, Total Count: ", epoch, acc/tot_count, tot_count)
    for dataloader in [dataloader_val]:
        acc, tot_count = 0.0, 0.001
        results_out = {}
        for idx, stuff in enumerate(dataloader):
            if stuff is None:
                continue
            g, bert_data, im_data, cap_data, title_data, ind_facs, KE_label, label, articleIDs = stuff
            pred_y_doc = model((g, bert_data, im_data, cap_data, title_data, ind_facs))
            pred_y_doc = torch.argmax(pred_y_doc, dim=-1)
            pred_y_doc = pred_y_doc.to("cpu")
            acc += (pred_y_doc.eq(label)).sum()
            tot_count += label.size()[0]
        print("Eval Accuracy: ", acc/tot_count, acc, tot_count)
        if mode == "inference":
            with open('results_output.json', 'w') as f:
                json.dump(results_out, f)
