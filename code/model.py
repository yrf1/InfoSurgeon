import torch, dgl
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from models.model_builder import AbsSummarizer

class SCAN(nn.Module):
    # Credits to: (Tan et al, 2020)
    def __init__(self):
        super(SCAN, self).__init__()
        self.cos = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, img_feats, cap_feats, cap_mask):
        img_feats = torch.reshape(img_feats, (-1, img_feats.size(-2), img_feats.size(-1)))
        combined_scores = []
        for i in range(cap_feats.size(1)):
            tmp_cap = cap_feats[:, i, :].unsqueeze(1).expand_as(img_feats)
            scores = self.cos(img_feats, tmp_cap)
            combined_scores.append(scores.unsqueeze(-1))
        combined_scores = torch.cat(combined_scores, dim=-1)
        combined_scores = self.softmax(combined_scores)
        img_feats = img_feats.unsqueeze(2).repeat(1, 1, combined_scores.size(2), 1)
        combined_scores = combined_scores.unsqueeze(-1).repeat(1, 1, 1, img_feats.size(-1))
        img_cap_reps = combined_scores * img_feats
        return torch.sum(img_cap_reps, dim=1)

class InfoSurgeon(nn.Module):
    def __init__(self, use_IE=False, use_gen_f=True, device="cpu"):
        super(InfoSurgeon, self).__init__()
        self.init_layer_edge = nn.Sequential(nn.Linear(300, 300), nn.ReLU())
        self.enc1 = nn.LSTM(300, 300, 2, bidirectional=True)
        self.enc2 = nn.LSTM(300, 300, 2, bidirectional=True)
        self.WW = nn.Linear(2*300, 2*300)
        self.fc = nn.Sequential(nn.Linear(1112+88, int(300/2)), \
                nn.ReLU(), \
                nn.Dropout(p=0.1))
        self.fc_attn = nn.Linear(int(300/2)+3+50,int(300/2)+3+50)
        self.classify_e = nn.Sequential(nn.Linear(306+2+2*49, 49), \
            nn.Dropout(p=0.1), \
            nn.ReLU(), nn.Linear(49,2))
        self.W_arts = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.W_imgs = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.W_cap = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.W = nn.Linear(512,50)
        self.bert = AbsSummarizer(device)
        self.scan = SCAN()
        self.classify_g = nn.Sequential(nn.Linear(512+3, 512), \
            nn.ReLU(), nn.Linear(512,56), \
            nn.ReLU(), nn.Linear(56,2))
        self.loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.use_gen_f, self.use_IE = use_gen_f, use_IE
        self.device = device
        self.edge_fc_global = nn.Linear(1536,512)
        self.edge_fc_local = nn.Linear(1536,512)
        self.edge_fc_local2global = nn.Linear(1712,512) 
        self.funcs = {}
        self.funcs['global_edge'] = (fn.copy_u('global_x', 'm'), fn.mean('m', 'h'))
        self.funcs['local_edge'] = (fn.copy_u('local_x', 'm'), fn.mean('m', 'h'))
    def edge_attention_global(self, edges):
        z2 = torch.cat([edges.src['global_x'], edges.data['global_x'], edges.dst['global_x']], dim=1)
        return {'e': F.leaky_relu(self.edge_fc_global(z2))}
    def edge_attention_local(self, edges):
        z2 = torch.cat([edges.src['local_x'], edges.data['local_x'], edges.dst['local_x']], dim=1)
        return {'e': F.leaky_relu(self.edge_fc_local(z2))}
    def edge_attention_local2global(self, edges):
        z2 = torch.cat([edges.src['h'], edges.data['local2global_x'], edges.dst['global_x']], dim=1)
        return {'e': F.leaky_relu(self.edge_fc_local2global(z2))}
    def message_func(self, edges):
        return {'e': edges.data['e']}
    def reduce_func(self, nodes):
        return {'h': torch.mean(nodes.mailbox['e'], dim=1)}
    def message_func_global2local(self, edges):
        return {'h': edges.src['h']}
    def reduce_func_global2local(self, nodes):
        return {'h_g2l': torch.mean(nodes.mailbox['h'], dim=1)}
    def final_message_func(self, edges):
        return {'score' : edges.data['score'], 'score_attn':edges.data['score_attn']}
    def final_reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['score_attn'], dim=1)
        return {'secondary_score' : torch.sum(alpha * nodes.mailbox['score'], dim=1)}
    def forward(self, stuff): 
        g, bert_data, img_data, cap_data, title_data, ind_facs = stuff
        src, tgt, segs, clss, mask_src, mask_tgt, mask_cls = bert_data
        if self.device == "cuda":
            src, tgt, segs, clss, mask_src, mask_tgt, mask_cls = src.cuda(), \
                    tgt.cuda(), segs.cuda(), clss.cuda(), mask_src.cuda(), mask_tgt.cuda(), mask_cls.cuda()
        bert_feat, _, _ = self.bert(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
        cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls = cap_data
        title_src, title_tgt, title_segs, title_clss, title_mask_src, title_mask_tgt, title_mask_cls = title_data
        if self.device == "cuda":
            cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls = cap_src.cuda(), \
                    cap_tgt.cuda(), cap_segs.cuda(), cap_clss.cuda(), cap_mask_src.cuda(), cap_mask_tgt.cuda(), cap_mask_cls.cuda()
            img_data, ind_facs = img_data.cuda(), ind_facs.cuda()
            title_src, title_tgt, title_segs, title_clss, title_mask_src, title_mask_tgt, title_mask_cls = \
                    title_src.cuda(), title_tgt.cuda(), title_segs.cuda(), title_clss.cuda(), title_mask_src.cuda(), title_mask_tgt.cuda(), title_mask_cls.cuda()
        cap_word_embeds, _, _ = self.bert(cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls)
        cap_word_embeds = self.W_cap(cap_word_embeds)
        cap_embeds = torch.sum(cap_word_embeds, dim=1)
        cap_tok_counts = torch.sum(cap_mask_src, dim=-1).unsqueeze(-1).expand_as(cap_embeds)
        cap_embeds /= (cap_tok_counts.cuda() if self.device=="cuda" else cap_tok_counts).float() 
        img_feats = self.W_imgs(img_data)
        img_cap_reps = self.scan(img_feats, cap_word_embeds, cap_mask_src)
        img_cap_reps = torch.sum(img_cap_reps, dim=1)
        cap_tok_counts = torch.sum(cap_mask_src, dim=-1).unsqueeze(-1).expand_as(img_cap_reps)
        img_cap_reps /= (cap_tok_counts.cuda() if self.device=="cuda" else cap_tok_counts).float()
        bert_feat = self.W_arts(F.relu(bert_feat))
        img_art_reps = self.scan(img_feats, bert_feat, mask_src)
        img_art_reps = torch.sum(img_art_reps, dim=1)
        art_tok_counts = torch.sum(mask_src, dim=-1).unsqueeze(-1).expand_as(img_art_reps)
        img_art_reps /= (art_tok_counts.cuda() if self.device=="cuda" else art_tok_counts).float()
        art_cap_reps = self.scan(cap_word_embeds.unsqueeze(1),bert_feat, mask_src)
        art_cap_reps = torch.sum(art_cap_reps, dim=1)
        art_cap_reps /= (art_tok_counts.cuda() if self.device=="cuda" else art_tok_counts).float()
        art_embeds = torch.sum(bert_feat, dim=1)
        num_toks = torch.sum(mask_src, dim=-1).cuda() if self.device=="cuda" else torch.sum(mask_src, dim=-1)
        art_embeds = art_embeds / num_toks.unsqueeze(-1).expand_as(art_embeds).float()
        img_embeds = torch.mean(img_feats, 2).squeeze(1)
        title_word_embeds, _, _ = self.bert(title_src, title_tgt, title_segs, title_clss, title_mask_src, title_mask_tgt, title_mask_cls)
        title_word_embeds = self.W_cap(title_word_embeds)
        title_embeds = torch.sum(title_word_embeds, dim = 1)
        title_tok_counts = torch.sum(title_mask_src, dim=-1).unsqueeze(-1).expand_as(title_embeds)
        title_embeds /= (title_tok_counts.cuda() if self.device=="cuda" else title_tok_counts).float()
        art_title_reps = self.scan(title_word_embeds.unsqueeze(1), bert_feat, mask_src)
        art_title_reps = torch.sum(art_title_reps, dim=1)
        art_title_reps /= (art_tok_counts.cuda() if self.device=="cuda" else art_tok_counts).float()
        # Reference: https://github.com/dmlc/dgl/blob/ac282a5e35ca7f8abdc8c14bde0d7db1305e2fe9/examples/pytorch/hgt/model.py
        for i in range(len(g)): 
            g[i].nodes['global_node'].data['global_x'] = torch.stack([art_embeds[i,:], img_embeds[i,:], cap_embeds[i,:], title_embeds[i]])
            g[i].edges['global_edge'].data['global_x'] = torch.stack([img_art_reps[i,:], art_cap_reps[i,:], img_cap_reps[i,:], art_title_reps[i,:]])
        g = dgl.batch(g)
        g.edges['local_edge'].data['local_x1'] = self.init_layer_edge(g.edges['local_edge'].data['local_x1'])
        g.edges['local_edge'].data['local_x1'] = self.enc1(g.edges['local_edge'].data['local_x1'])[0][:,-1,:].unsqueeze(1)
        g.edges['local_edge'].data['local_x2'] = self.init_layer_edge(g.edges['local_edge'].data['local_x2'])
        g.edges['local_edge'].data['local_x2'] = self.enc2(g.edges['local_edge'].data['local_x2'])[0]
        g.edges['local_edge'].data['local_x'] = g.edges['local_edge'].data['local_x1'].expand_as(g.edges['local_edge'].data['local_x2'])*\
                                                                                                 g.edges['local_edge'].data['local_x2']
        g.edges['local_edge'].data['local_x'] = self.relu(self.WW(g.edges['local_edge'].data['local_x']))
        g.edges['local_edge'].data['e'] = torch.max(g.edges['local_edge'].data['local_x'], dim=1)[0]
        g.edges['local2global_edge'].data['local2global_x'] = self.init_layer_edge(g.edges['local2global_edge'].data['local2global_x'])
        g.edges['local2global_edge'].data['local2global_x'] = self.enc1(g.edges['local2global_edge'].data['local2global_x'])[0][:,-1,:]
        g.update_all(self.message_func, self.reduce_func, etype="local_edge")
        g.apply_edges(self.edge_attention_local2global, etype='local2global_edge')
        g.apply_edges(self.edge_attention_global, etype='global_edge')
        g.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in ["global_edge","local2global_edge"]}, cross_reducer='mean')
        overall_feats = torch.cat((dgl.max_nodes(g, 'h', ntype="global_node"), ind_facs), dim=-1)
        g.multi_update_all({etype : (self.message_func_global2local, self.reduce_func_global2local) \
                            for etype in ["global2local_edge"]}, cross_reducer='mean')
        g.edges['local_edge'].data['e'] = torch.cat((g.edges['local_edge'].data['e'],g.edges['local_edge'].data['local_x1'].squeeze(1)),dim=1)
        g.edges['local_edge'].data['score'] = self.relu(self.fc(g.edges['local_edge'].data['e'])) 
        g.apply_edges(lambda edges: {'score': torch.cat((g.edges['local_edge'].data['score'],g.edges['local_edge'].data['ind_feats'],\
                self.relu(self.W(edges.src['h_g2l']))),dim=1)}, etype="local_edge")
        g.edges['local_edge'].data['score_attn'] = F.leaky_relu(self.fc_attn(g.edges['local_edge'].data['score']))
        g.update_all(self.final_message_func, self.final_reduce_func, etype="local_edge") 
        g.apply_edges(lambda edges: {'score' : torch.cat((edges.data["score"],edges.src['secondary_score']), dim=1)}, etype="local_edge")
        g.edges['local_edge'].data['score'] = self.classify_e(g.edges['local_edge'].data['score'])
        return self.classify_g(F.relu(overall_feats)), g
    def compute_loss(self, pred_y, true_y):
        if self.use_IE:
            pred_y = pred_y.squeeze(0)
        real_loss = self.loss(pred_y[:,0], 1-true_y)
        fake_loss = self.loss(pred_y[:,1], true_y)
        return real_loss + fake_loss
