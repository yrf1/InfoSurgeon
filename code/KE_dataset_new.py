import os
import re
import dgl
import json
import torch 
import bisect
import pickle
import random
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from torch.utils.data import Dataset

class KEDataset(Dataset):
    def __init__(self, data, trainValTest="train", use_IE=True, use_gen_f=True):
        self.glove_embed, self.data_dir = data
        self.all_data, bert_data, cap_bert_data, title_bert_data = {}, [], [], []
        self.mapping = pd.read_csv(self.data_dir[0]+"mapping.csv", header=None)
        self.list_of_articleIDs, self.use_IE, self.use_gen_f = [], use_IE, use_gen_f
        self.list_of_articleIDs = self.mapping[self.mapping[3]==trainValTest][0].values
        for data_dir in self.data_dir:
            bert_data_split = torch.load(data_dir+"/BERT_DATA_PATH/all.bert.pt")
            bert_data.extend(bert_data_split)
            cap_bert_data.extend(torch.load(data_dir+"/caption/BERT_DATA_PATH/all.bert.pt"))
            title_bert_data.extend(torch.load(data_dir+"/title/BERT_DATA_PATH/all.bert.pt"))
        self.bert_data, self.cap_bert_data, self.title_bert_data = {}, {}, {}
        for data in bert_data:
            data["name"] = data["name"].replace(".","_")
            self.bert_data[data["name"]] = data
        for data in cap_bert_data:
            self.cap_bert_data[data["name"]] = data
        for data in title_bert_data:
            self.title_bert_data[data["name"]] = data
        with open(self.data_dir[0]+"/IE_results.pkl", "rb") as f:
            self.KG_data = pickle.load(f)
        self.KB = pd.read_csv("NLP_toolbox/YiBase.csv", header=None)
        with open(data_dir+"ind_facs.json", "r") as f:
            self.ind_facs = json.load(f)
        self.list_of_articleIDs = [x for x in self.list_of_articleIDs if str(x) in list(self.bert_data.keys())]
        print(len(self.list_of_articleIDs), self.list_of_articleIDs[0], type(self.list_of_articleIDs[0]))
        self.glove_embed_avg = []
        for k, v in self.glove_embed.items():
            if len(v) > 0:
                self.glove_embed_avg.append(v)
        self.glove_embed_avg = np.mean(np.array(self.glove_embed_avg),axis=0)
        
    def __len__(self):
        return len(self.list_of_articleIDs)

    def __getitem__(self, idx):
        articleID = self.list_of_articleIDs[idx]
        try:
            return self.get_gen_f(articleID) 
        except:
            print(idx)
            return

    def get_gen_f(self, articleID):
        im_data, cap_data = [], []
        label = self.mapping[self.mapping[0]==articleID][2].values[0]
        articleID = str(articleID)
        src, tgt, segs, clss = self.preprocess(self.bert_data[articleID])
        for i in range(1): #extendible to other img,cap pairs 
            if articleID+'_cap_'+str(i) in self.cap_bert_data:
                im_data_feats = np.zeros((36, 2048))
                for data_dir in self.data_dir:
                    if os.path.exists(data_dir+'bottom_up_attention/'+articleID+'_img_'+str(i)+'.npz'):
                        im_data_feats = np.load(data_dir+'bottom_up_attention/'+articleID+'_img_'+str(i)+'.npz')["x"]
                im_data.append(im_data_feats)
                cap_data.append(self.preprocess(self.cap_bert_data[articleID+'_cap_'+str(i)]))
        im_data = [np.zeros((36, 2048))] if len(im_data)==0 else im_data
        cap_data = None if len(cap_data)==0 else cap_data
        title_data = self.preprocess(self.title_bert_data[articleID+"_title"]) \
                if articleID+"_title" in self.title_bert_data \
                else self.preprocess(self.title_bert_data[articleID+"_metadata"]) \
                if articleID+"_metadata" in self.title_bert_data else cap_data[0] 
        cap_data = [title_data] if cap_data==None else cap_data
        local_edges, local_nfeats, local_efeats1, local_efeats2, local_ind_feats, local_train_flag, KE_label, local2global_edges = self.get_local_f(articleID)
        g = {('global_node', 'global_edge', 'global_node'): (torch.tensor([0,0,1,0]), torch.tensor([1,2,2,3])), \
             ('local_node', 'local_edge', 'local_node'): (torch.tensor([x[0] for x in local_edges]), torch.tensor([x[1] for x in local_edges])), \
             ('global_node', 'global2local_edge', 'local_node'): (torch.tensor([x[1] for x in local2global_edges]), torch.tensor([x[0] for x in local2global_edges])), \
             ('local_node', 'local2global_edge', 'global_node'): (torch.tensor([x[0] for x in local2global_edges]), torch.tensor([x[1] for x in local2global_edges]))}
        g = dgl.heterograph(g)
        g.nodes['local_node'].data['local_x'] = torch.tensor(local_nfeats).float()
        g.edges['local_edge'].data['local_x1'] = torch.tensor(local_efeats1).float()
        g.edges['local_edge'].data['local_x2'] = torch.tensor(local_efeats2).float()
        g.edges['local_edge'].data['ind_feats'] = torch.tensor(local_ind_feats).float()
        g.edges['local_edge'].data['local_train_flag'] = torch.tensor(local_train_flag).float()
        if "VOA" in self.data_dir[0] or "VOA" in self.data_dir:
            g.edges['local_edge'].data['labels'] = torch.tensor(KE_label).float()
        g.edges['local_edge'].data['doc_label'] = torch.tensor([label]*g.edges['local_edge'].data['labels'].size()[0]).float()
        g.edges['local2global_edge'].data['local2global_x'] = torch.tensor(len(local2global_edges)*[self.str2idx("is part of")[0]]).float()
        ind_fac = self.ind_facs[articleID]["0"] if "0" in self.ind_facs[articleID] else [0, 0, 0]
        return src, tgt, segs, clss, im_data, cap_data, title_data, ind_fac, KE_label, label, articleID, g

    def preprocess(self, ex):
        max_pos, max_tgt_len = 512, 140
        src = ex['src']
        tgt = ex['tgt'][:max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        end_id = [src[-1]]
        tmp = src[:-1][:max_pos - 1] + end_id
        src = src[:-1][:max_pos - 1] + end_id
        segs = segs[:max_pos]
        max_sent_id = bisect.bisect_left(clss, max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        return src, tgt, segs, clss 

    def get_local_f(self, artID):
        art_triplets, modality, _ = self.KG_data[artID]
        if "VOA" in self.data_dir[0] or "VOA" in self.data_di:
            art_triplet_labels = self.KG_data[artID][-1]
        edges, node_feats = [], []
        edge_feats, node_lookup = [], {}
        edge_feats2, train_flag, ind_feats = [], [], []
        KE_label = []
        local2global_edges = []
        if ("VOA" in self.data_dir[0] or "VOA" in self.data_dir) and 1 in art_triplet_labels:
            first_neg, first_pos = [], []
            for i in range(len(art_triplet_labels)):
                if art_triplet_labels[i] == 0:
                    first_neg.append(i)
                if art_triplet_labels[i] == 1:
                    first_pos.append(i)
            first_neg, sec_neg, first_pos = random.choice(first_neg), random.choice(first_neg), random.choice(first_pos)
        else:
            first_neg, first_pos = 0, 0
        for i, trip in enumerate(art_triplets):
            train_ctr_val = (i == first_neg or i == first_pos or \
                    trip[0] == art_triplets[first_neg][0] or trip[2] == art_triplets[first_neg][0] or \
                    trip[0] == art_triplets[first_pos][0] or trip[2] == art_triplets[first_pos][0] or \
                    trip[0] == art_triplets[first_neg][1] or trip[2] == art_triplets[first_neg][1] or \
                    trip[0] == art_triplets[first_pos][1] or trip[2] == art_triplets[first_pos][1]) and 1 in art_triplet_labels
            node_num1, node_num2 = trip[0], trip[2]
            if node_num1[0] not in node_lookup:
                node_lookup[node_num1[0]] = len(node_lookup)
                node_feats.append(self.str2idx(trip[0][4], 10)[0])
            if node_num2[0] not in node_lookup:
                node_lookup[node_num2[0]] = len(node_lookup)
                node_feats.append(self.str2idx(trip[2][4], 10)[0])
            edge_feat, edge_len = self.str2idx(trip[0][4]+","+\
                                               self.shorten_relation(trip[1][0])+\
                                               ","+trip[2][4])
            edge_feats.append(edge_feat)
            edge_feat2, edge_len2 = self.str2idx(self.getBK(trip[0],trip[2]), 512)
            if trip[0][1][:2] == "m." and trip[2][1][:2] == "m.":
                ind_feats.append(modality[i]+[edge_len2, 0, 0])
            elif trip[0][1][:2] == "m." or trip[2][1][:2] == "m.":
                ind_feats.append(modality[i]+[0, edge_len2, 0])
            else:
                ind_feats.append(modality[i]+[0, 0, edge_len2])
            edge_feats2.append(edge_feat2)
            edges.append((node_lookup[node_num1[0]], node_lookup[node_num2[0]]))
            train_flag.append(train_ctr_val)
            if "VOA" in self.data_dir[0] or "VOA" in self.data_dir:
                KE_label.append(art_triplet_labels[i])
            if modality[i][0] == 1 or 1 not in modality[i]:
                local2global_edges.append((node_lookup[node_num1[0]],0))
            if modality[i][1] == 1:
                local2global_edges.append((node_lookup[node_num1[0]],1))
            if modality[i][2] == 1:
                local2global_edges.append((node_lookup[node_num1[0]],2))
            if modality[i][3] == 1:
                local2global_edges.append((node_lookup[node_num1[0]],3))
        return edges, node_feats, edge_feats, edge_feats2, ind_feats, train_flag, KE_label, local2global_edges

    def shorten_relation(self, r):
        if "-" not in r and "actual" not in r:
            r = r.split(".")[-1]
        else:
            last_chunk = r.replace(".actual","").split(".")[-1]
            last_chunk = last_chunk.replace("_",".")
            last_chunk = last_chunk.replace("-"+last_chunk.split(".")[0]+".","-")
            r = last_chunk if last_chunk != "" else r
        return r

    def str2idx(self, s, pad_len=23):
        embed = []
        for token in re.split("(['_\-.,<> ])", s.strip()):
            for tok in re.sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', token).split(" "):
                tok = tok.strip()
                if len(tok) > 0:
                    if tok in self.glove_embed:
                        token_embed = self.glove_embed[tok]
                    else:
                        token_embed = self.glove_embed_avg
                    embed.append(token_embed.tolist())
        embed_len = len(embed)
        for _ in range(pad_len-embed_len):
            embed.append(np.zeros(300))
        if embed_len > pad_len:
            embed, embed_len = embed[:pad_len], pad_len
        return embed, embed_len

    def getBK(self, n1, n2):
        KB = ""
        if n1[1][:2] == "m.":
            try:
                fbID_1,wikiID_1,url1,txt1,html1  = self.KB.loc[self.KB[0] == n1[1]].values[0]
                if n2[1][:2] == "m.":
                    fbID_2,wikiID_2,url2,txt2,html2  = self.KB.loc[self.KB[0] == n2[1]].values[0]
                    if url2.lstrip("https://en.wikipedia.org/") in html1:
                        soup = BeautifulSoup(html1)
                        for p in soup.find_all(href=url2.replace("https://en.wikipedia.org","")):
                            p = p.find_parent('p')
                            if p:
                                p = p.text
                                KB += re.sub("[\[].*?[\]]", "", p)
                else:
                    for p in txt1.split("\n"):
                        if n2[4] in p:
                            KB += p
            except:
                pass
        if KB == "" and n2[1][:2] == "m.":
            try:
                fbID_2,wikiID_2,url2,txt2,html2  = self.KB.loc[self.KB[0] == n2[1]].values[0]
                for p in txt2.split("\n"):
                    if n1[4] in p:
                        KB += p
            except:
                pass
        return KB
