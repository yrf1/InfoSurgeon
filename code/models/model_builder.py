import copy
import sys
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class AbsSummarizer(nn.Module):
    def __init__(self, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        large, temp_dir, finetune_bert, share_embed = False, "../temp", True, False
        enc_hidden_size, enc_layers, enc_ff_size, enc_dropout = 512, 6, 512, 0.2
        max_pos, dec_layers, dec_hidden_size, dec_heads, dec_ff_size, dec_dropout = 512, 6, 768, 8, 2048, 0.2
        #self.args = args
        self.device = device
        self.bert = Bert(large, temp_dir, finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        #if (args.encoder == 'baseline'):
        #    bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=enc_hidden_size,
        #                             num_hidden_layers=enc_layers, num_attention_heads=8,
        #                             intermediate_size=enc_ff_size,
        #                             hidden_dropout_prob=enc_dropout,
        #                             attention_probs_dropout_prob=enc_dropout)
        #    self.bert.model = BertModel(bert_config)

        ##if(args.max_pos>512):
        #    my_pos_embeddings = nn.Embedding(max_pos, self.bert.model.config.hidden_size)
        #    my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        #    my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_pos-512,1)
        #    self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        #if (share_emb):
        #    tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            dec_layers, dec_hidden_size, heads=dec_heads,
            d_ff=dec_ff_size, dropout=dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            #if(args.use_bert_emb):
            #    tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
            #    tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
            #    self.decoder.embeddings = tgt_embeddings
            #    self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)

        '''print('src: ' + str(src.shape))
        print('tgt: ' + str(tgt.shape))
        print('segs: ' + str(segs.shape))
        print('mask_src: ' + str(mask_src.shape))
        print('top_vec: ' + str(top_vec.shape))
        print('decoder_outputs: ' + str(decoder_outputs.shape))
        sys.exit()'''

        '''print('src: ' + str(src.shape))
        print('segs: ' + str(segs.shape))
        print('mask_src: ' + str(mask_src.shape))
        print('top_vec: ' + str(top_vec.shape))
        #print('dec_state: ' + str(dec_state.shape))
        print('decoder_outputs: ' + str(decoder_outputs.shape))
        return'''

        return top_vec, decoder_outputs, None
