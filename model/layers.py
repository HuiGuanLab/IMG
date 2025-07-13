
from email.mime import audio
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
def spatial_broadcast(slots, max_pos_len):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = torch.reshape(slots, [-1, slots.shape[-1]]).unsqueeze(1)

  grid = slots.repeat(1,max_pos_len, 1)
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid
def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = torch.reshape(x, [batch_size, -1] + list(x.shape)[1:])
#   channels, masks = torch.split(unstacked, [num_channels,1], dim=-1)
#   print(mask.shape)
#   print(channels.shape)
#   exit(0)
  return unstacked

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(Conv1D(in_dim=n, out_dim=k, kernel_size=1, stride=1, padding=0, bias=True) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def apply_on_sequence(layer, inp):
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output



class DownSample(nn.Module):
    def __init__(self, in_dim,out_dim,drop_rate):
        super(DownSample, self).__init__()  
        self.conv = Conv1D(in_dim=in_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout=nn.Dropout(p=drop_rate)
    def forward(self, x):
        # compute logits
        x = self.dropout(x)
        x = self.conv(x)
        # x = self.dropout(x)
        return x




def get_mean_feature(feature,mask):
    length = torch.sum(mask,dim=1)
    mean_feature=torch.zeros((feature.shape[0],feature.shape[2]),dtype=torch.float32).to(feature.device)
    for i in range(feature.shape[0]):
        mean_feature[i]=torch.mean(feature[i][:int(length[i])],dim=0)
    return mean_feature

    
class SlotAttention(nn.Module):
    def __init__(self, iters, num_slots, dim,eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.to_k = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.to_v = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = dim

        self.mlp = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace = True),
            Conv1D(in_dim=hidden_dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs,mask, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots.masked_fill_(mask.unsqueeze(1)==0, -1e30)
            attn = dots.softmax(dim=-1)
            # attn = attn / attn.sum(dim=-1, keepdim=True)


            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class ParameterPredictor(nn.Module):
    def __init__(self, dim):
        super(ParameterPredictor, self).__init__()
        self.pred_proj=nn.Linear(2 * dim,1)

        self.weighted_poolv = WeightedPool(dim=dim)
        self.weighted_poola = WeightedPool(dim=dim)
        # self.cls_v = nn.Parameter(torch.zeros(1, 1, dim),requires_grad=True)
        # self.cls_a = nn.Parameter(torch.zeros(1, 1, dim),requires_grad=True)
        # self.attention_blockv = GlobalAttention(idim=dim,
        #                                      odim=dim,
        #                                      nheads=1,
        #                                      dp=0.1)
        # self.attention_blocka = GlobalAttention(idim=dim,
        #                                      odim=dim,
        #                                      nheads=1,
        #                                      dp=0.1)
        # self.attention_blockv2 = GlobalAttention(idim=dim,
        #                                      odim=dim,
        #                                      nheads=1,
        #                                      dp=0.1)
        # self.attention_blocka2 = GlobalAttention(idim=dim,
        #                                      odim=dim,
        #                                      nheads=1,
        #                                      dp=0.1)
        # self.attention_blocka =  MultiHeadAttentionBlock(dim=dim, num_heads=8, drop_rate=0.2)
        # self.attention_blockv =  MultiHeadAttentionBlock(dim=dim, num_heads=8, drop_rate=0.2)

    def forward(self, video_features,audio_features,va_mask,p_warm):
        audio_features = self.weighted_poola(audio_features,va_mask)
        video_features = self.weighted_poolv(video_features,va_mask)
        # print(audio_features.shape)
        # exit(0)
        # print(audio_features.shape)
        # print(video_features.shape)
        # B=video_features.shape[0]
        # audio_features,_=self.attention_blocka(audio_features,va_mask)
        # video_features,_=self.attention_blockv(video_features,va_mask)
        # audio_features = get_mean_feature(audio_features,va_mask)
        # video_features = get_mean_feature(video_features,va_mask)
        # audio_features=torch.cat((self.cls_a.repeat(B,1,1),audio_features),dim=1)
        # video_features=torch.cat((self.cls_v.repeat(B,1,1),video_features),dim=1)
        # add_mask=torch.ones((B,1),dtype=torch.float32).to(audio_features.device)
        # va_mask=torch.cat((add_mask,va_mask),dim=-1)
        # audio_features,_ = self.attention_blocka(audio_features,mask=va_mask)
        # video_features,_ = self.attention_blockv(video_features,mask=va_mask)
        # audio_features,_ = self.attention_blocka2(audio_features,mask=va_mask)
        # video_features,_ = self.attention_blockv2(video_features,mask=va_mask)

        # audio_features = get_mean_feature(audio_features,va_mask)
        # video_features = get_mean_feature(video_features,va_mask)

        fuse = torch.cat([audio_features,video_features],dim=-1)
        global_audio=nn.Sigmoid()(self.pred_proj(fuse)).unsqueeze(-1)
        

        true_global=0.5*(1-p_warm)+p_warm*global_audio

        return true_global,global_audio




class ParameterFusion(nn.Module):
    def __init__(self, dim):
        super(ParameterFusion, self).__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
    
    def forward(self, video_features,audio_features,ap,mask):
            
        fusion = (1-ap)*self.layer_norm1(video_features)+ap*self.layer_norm2(audio_features)
        

        
        return fusion



class FusionLayerAV(nn.Module):
    def __init__(self, dim):
        super(FusionLayerAV, self).__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        # self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, video_features,audio_features,alpha_av):
        fusion=(1-alpha_av)*self.layer_norm1(video_features)+alpha_av*self.layer_norm2(audio_features)
        return fusion





class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, dim, 2).float()/dim)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, dim, 2).float()/dim)
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss,loss_per_location* mask
        
class GlobalAttention(nn.Module):
    def __init__(self, idim, odim, nheads, dp):
        super(GlobalAttention, self).__init__()
        self.idim = idim
        self.odim = odim
        self.nheads = nheads
        
        self.use_bias = True
        self.use_local_mask = False
        
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dp)

    def forward(self, m_feats, mask):
        mask = mask.float()
        B, nseg = mask.size()

        m_k = self.v_lin(self.drop(m_feats))
        m_trans = self.c_lin(self.drop(m_feats))
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i]

            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9)
            m2m_w = F.softmax(m2m, dim=2)
            w_list.append(m2m_w)

            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)
        
        updated_m = self.drop(m_feats+r)
        return updated_m, torch.stack(w_list, dim=1)

 
class SequentialQueryAttention(nn.Module):
    def __init__(self, nse, qdim):
        super(SequentialQueryAttention, self).__init__()
        self.nse = nse
        self.qdim = qdim
        self.global_emb_fn = nn.ModuleList(
                [nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2*self.qdim, self.qdim),
            nn.ReLU()
        ])
        self.att_fn = Attention(kdim=self.qdim, cdim=self.qdim, 
                                att_hdim=self.qdim // 2, drop_p=0.0)

    def forward(self, q_feats, w_feats, w_mask=None):
        B = w_feats.size(0)
        prev_se = w_feats.new_zeros(B, self.qdim)
        se_feats, se_attw = [], []
        
        for n in range(self.nse):
            q_n = self.global_emb_fn[n](q_feats)
            g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1))
            att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w)

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)



class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)

class Conv1DTranspose(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1DTranspose, self).__init__()
        self.conv1d = nn.ConvTranspose1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)

class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)
        return char_output.permute(0, 2, 1)


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        self.linear = Conv1D(in_dim=word_dim + 100, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)
        char_emb = self.char_emb(char_ids)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.linear(emb)
        return emb


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)
        return output


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)
            output = output.transpose(1, 2)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual
        return output

class CrossMultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(CrossMultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def get_attention_mask(self, mask_query, mask):
        attention_mask = torch.matmul(mask_query.transpose(-1, -2), mask)
        return attention_mask


    def forward(self, x1,x2, mask1=None,mask2=None):
        output1 = self.layer_norm1(x1)  # (batch_size, seq_len, dim)
        output1 = self.dropout(output1)

        output2 = self.layer_norm2(x2)  # (batch_size, seq_len, dim)
        output2 = self.dropout(output2)
        # multi-head attention layer
        query = self.transpose_for_scores(self.query(output1))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output2))
        value = self.transpose_for_scores(self.value(output2))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        mask = self.get_attention_mask(mask1, mask2)

        mask = mask.type(torch.float32)
        mask = (1.0 - mask.unsqueeze(1)) * (-1e30)  # (N, 1, Lq, L)
        attention_scores = attention_scores+mask
        oattention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, seq_len, seq_len)

        attention_probs = self.dropout(oattention_probs)
        value = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x1
        output = self.layer_norm3(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output,oattention_probs



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)
        output = self.dropout(output)
        query = self.transpose_for_scores(self.query(output))
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output



# class SelfAndCrossAttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, drop_rate):
#         super(SelfAndCrossAttentionBlock, self).__init__()
#         assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
#         self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

#         self.queryc = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.keyc = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.valuec = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

#         self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
#         self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)

#         self.layer_norm4 = nn.LayerNorm(dim, eps=1e-6)
#         self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

#     def get_attention_mask(self, mask_query, mask):
#         attention_mask = torch.matmul(mask_query.transpose(-1, -2), mask)
#         return attention_mask

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     @staticmethod
#     def combine_last_two_dim(x):
#         old_shape = list(x.size())
#         new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
#         return x.reshape(shape=new_shape)

#     def forward(self, x1,x2, mask1=None,mask2=None):
#         output1 = self.layer_norm1(x1)
#         output1 = self.dropout(output1)
#         query1 = self.transpose_for_scores(self.query(output1))
#         key1 = self.transpose_for_scores(self.key(output1))
#         value1 = self.transpose_for_scores(self.value(output1))
#         attention_scores1 = torch.matmul(query1, key1.transpose(-1, -2))

#         attention_scores1 = attention_scores1 / math.sqrt(self.head_size)
#         if mask1 is not None:
#             mask1 = mask1.unsqueeze(2)
#             attention_scores1 = mask_logits(attention_scores1, mask1)
#         attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
#         attention_probs1 = self.dropout(attention_probs1)

#         output2 = self.layer_norm2(x1)
#         output2 = self.dropout(output2)

#         output3 = self.layer_norm3(x2)
#         output3 = self.dropout(output3)      
#         query2 = self.transpose_for_scores(self.queryc(output2))
#         key2 = self.transpose_for_scores(self.keyc(output3))
#         value2 = self.transpose_for_scores(self.valuec(output3))
#         attention_scores2 = torch.matmul(query2, key2.transpose(-1, -2))

#         attention_scores2 = attention_scores2 / math.sqrt(self.head_size)
#         mask = self.get_attention_mask(mask1.squeeze(2), mask2)
#         mask = mask.type(torch.float32)

#         mask = (1.0 - mask.unsqueeze(1)) * (-1e30)  # (N, 1, Lq, L)
#         attention_scores2 = attention_scores2+mask
#         attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)  # (batch_size, num_heads, seq_len, seq_len)
#         attention_probs2 = self.dropout(attention_probs2)




   
#         value1 = torch.matmul(attention_probs1, value1)
#         value2 = torch.matmul(attention_probs2, value2)

#         value=value1+value2

#         value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))
#         output = self.dropout(value)
#         residual = output + x1 
#         output = self.layer_norm4(residual)
#         output = self.dropout(output)
#         output = self.out_layer(output)
#         output = self.dropout(output) + residual
#         return output

class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        # self.pe=PositionalEncoding(dim=dim,max_len=max_pos_len)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        # features = self.pe(x)
        features =  x + self.pos_embedding(x)
        features = self.conv_block(features)
        features = self.attention_block(features, mask=mask)
        return features



class Encoder(nn.Module):
    def __init__(self,max_len, dim, num_heads, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(Encoder, self).__init__()
        self.pe=PositionalEncoding(dim=dim,max_len=max_len)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        x=self.pe(x)
        features = self.conv_block(x)
        features = self.attention_block(features, mask=mask)
        return features

class ConvBlockEncoder(nn.Module):
    def __init__(self, dim, num_heads,max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(ConvBlockEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)
        features = self.conv_block(features)
        features = self.attention_block(features, mask=mask)
        return features

class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)
        score_ = nn.Softmax(dim=2)(score if q_mask == None else mask_logits(score, q_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2q = torch.matmul(score_, query)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class VideoTextConcatenate(nn.Module):
    def __init__(self, dim):
        super(VideoTextConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output




class PConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, drop_rate=0.0):
        super(PConditionedPredictor, self).__init__()  
        self.encoder1 = nn.GRU( dim,dim//2,num_layers=3,batch_first=True,bidirectional=True)
        # self.encoder = FeatureEncoder(dim=3*dim, num_heads=num_heads, kernel_size=7, num_layers=4,
        #                                   max_pos_len=max_pos_len, drop_rate=drop_rate)

        self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2*dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2*dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):


        length=torch.sum(mask,dim=1).cpu().int()
        start_features=pack_padded_sequence(x,length,batch_first=True,enforce_sorted=False)
        start_features,_ = self.encoder1(start_features)
        start_features=pad_packed_sequence(start_features,batch_first=True)[0]

        end_features=pack_padded_sequence(start_features,length,batch_first=True,enforce_sorted=False)
        end_features,_ = self.encoder1(end_features)
        end_features=pad_packed_sequence(end_features,batch_first=True)[0]

        start_features = self.start_layer_norm(start_features)
        end_features = self.end_layer_norm(end_features)
        
        start_features = self.start_block(torch.cat([start_features, x], dim=2))
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='none')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='none')(end_logits, end_labels)
        return start_loss + end_loss

    @staticmethod
    def new_compute_distillation_loss(student_start, student_end,  teacher_start, teacher_end):
        T = 2
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_start/T, dim=1),
                                F.softmax(teacher_start.detach()/T, dim=1)) * T * T 
        
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_end/T, dim=1),
                                F.softmax(teacher_end.detach()/T, dim=1)) * T * T

    
        return torch.sum(KD_loss,dim=-1)
    @staticmethod
    def new_compute_selective_distillation_loss(student_start, student_end,  teacher_start, teacher_end,matrix):
        T = 2
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_start/T, dim=1),
                                F.softmax(teacher_start.detach()/T, dim=1)) * T * T 
        
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_end/T, dim=1),
                                F.softmax(teacher_end.detach()/T, dim=1)) * T * T
        
        KD_loss=torch.sum(KD_loss,dim=-1)

        length=torch.sum(matrix)

        KD_loss=torch.sum(KD_loss*matrix)/(length+1e-6)

        return KD_loss

class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, drop_rate=0.0):
        super(ConditionedPredictor, self).__init__()  
        # self.encoder1 = nn.GRU( dim,dim//2,num_layers=3,batch_first=True,bidirectional=True)
        self.encoder = FeatureEncoder(dim=dim, num_heads=num_heads, kernel_size=7, num_layers=4,
                                          max_pos_len=max_pos_len, drop_rate=drop_rate)

        self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):


        start_features = self.encoder(x, mask)
        end_features = self.encoder(start_features, mask)
        start_features = self.start_layer_norm(start_features)
        end_features = self.end_layer_norm(end_features)
        
        start_features = self.start_block(torch.cat([start_features, x], dim=2))
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='none')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='none')(end_logits, end_labels)
        return start_loss + end_loss

    @staticmethod
    def new_compute_distillation_loss(student_start, student_end,  teacher_start, teacher_end):
        T = 2
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_start/T, dim=1),
                                F.softmax(teacher_start.detach()/T, dim=1)) * T * T 
        
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_end/T, dim=1),
                                F.softmax(teacher_end.detach()/T, dim=1)) * T * T

    
        return torch.sum(KD_loss,dim=-1)
    @staticmethod
    def new_compute_selective_distillation_loss(student_start, student_end,  teacher_start, teacher_end,matrix):
        T = 2
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_start/T, dim=1),
                                F.softmax(teacher_start.detach()/T, dim=1)) * T * T 
        
        KD_loss += nn.KLDivLoss(reduction='none')(F.log_softmax(student_end/T, dim=1),
                                F.softmax(teacher_end.detach()/T, dim=1)) * T * T
        
        KD_loss=torch.sum(KD_loss,dim=-1)

        length=torch.sum(matrix)

        KD_loss=torch.sum(KD_loss*matrix)/(length+1e-6)

        return KD_loss

class LocalAudioFusion(nn.Module):
    def __init__(self, dim,num_heads,  drop_rate=0.0):
        super(LocalAudioFusion, self).__init__()  
        self.conv_blocka = nn.ModuleList([nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=i, groups=dim,padding=i//2, bias=True) for i in range(1,7,2)])
        self.conv_blockv = nn.ModuleList([nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=i, groups=dim,padding=i//2, bias=True) for i in range(1,7,2)])
        
        self.downsamplea=DownSample(in_dim =dim*3,out_dim=dim,drop_rate=drop_rate)
        self.downsamplev=DownSample(in_dim =dim*3,out_dim=dim,drop_rate=drop_rate)

        # self.cross_transformer_a =  CrossMultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        # self.cross_transformer_v =  CrossMultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

        self.fusion= ParameterFusion(dim=dim)

    def forward(self, v,a,ap,va_mask):
        la = a.transpose(1, 2)
        la = [F.relu(conv_block(la)) for conv_block in self.conv_blocka]
        la = torch.cat(la,1)
        la = la.transpose(1, 2)
        la = self.downsamplea(la)

        lv = v.transpose(1, 2)
        lv = [F.relu(conv_block(lv)) for conv_block in self.conv_blockv]
        lv = torch.cat(lv,1)
        lv = lv.transpose(1, 2)
        lv = self.downsamplev(lv)

        av = self.fusion(lv,la,ap,va_mask)

        return av
class EventAudioFusion(nn.Module):
    def __init__(self, dim,num_heads,event_num,slot_layers,  drop_rate=0.0):
        super(EventAudioFusion, self).__init__()  
        self.event_num=event_num
        self.slot_attentiona =  SlotAttention(slot_layers, event_num, dim)
        self.slot_attentionv =  SlotAttention(slot_layers, event_num, dim)

        self.cross_transformer_a =  CrossMultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        self.cross_transformer_v =  CrossMultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        
        self.fusion= ParameterFusion(dim=dim)


    def forward(self, v,a,ap,va_mask):
        
        even_a = self.slot_attentiona(a, va_mask)
        even_v = self.slot_attentionv(v, va_mask)

        newa_mask=torch.ones(v.shape[0],self.event_num).to(va_mask.device)

        na,_ = self.cross_transformer_a(a,even_a,va_mask.unsqueeze(1),newa_mask.unsqueeze(1))
        nv,_ = self.cross_transformer_v(v,even_v,va_mask.unsqueeze(1),newa_mask.unsqueeze(1))

        av=self.fusion(nv,na,ap,va_mask)
        
        return av

class Simple_GatedAttention(nn.Module):
    def __init__(self, embed_dim):
        super(Simple_GatedAttention, self).__init__()
        self.embed_dim = embed_dim

        # 门控单元
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # 拼接后的维度是 embed_dim * 2
             nn.Sigmoid() # 门控权重
        )

        # # 可选的线性变换
        self.linear = nn.Linear(embed_dim, embed_dim)
        # self.linear = DownSample(in_dim =embed_dim,out_dim=embed_dim,drop_rate=0.2)

    def forward(self, global_token, local_tokens):
        # global_token: [batch_size, 1, embed_dim]
        # local_tokens: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = local_tokens.shape

        # 将全局信息复制到每个局部 token
        global_token_expanded = global_token.expand(-1, seq_len, -1) # [batch_size, seq_len, embed_dim]

        # 拼接全局信息和局部信息
        concatenated = torch.cat([global_token_expanded, local_tokens], dim=-1) # [batch_size, seq_len, embed_dim * 2]

        # 计算门控权重
        gate_weights = self.gate(concatenated) # [batch_size, seq_len, embed_dim]

        # 加权融合
        fused = gate_weights * global_token_expanded + (1 - gate_weights) * local_tokens # [batch_size, seq_len, embed_dim]

        # # 可选的线性变换
        output = self.linear(fused) # [batch_size, seq_len, embed_dim]

        return output

class GlobalAudioFusion(nn.Module):
    def __init__(self, dim,  num_heads,  drop_rate=0.0):
        super(GlobalAudioFusion, self).__init__()  
        self.global_audio = WeightedPool(dim = dim)
        self.global_video = WeightedPool(dim = dim)

        self.downsamplea=DownSample(in_dim=2*dim,out_dim=dim,drop_rate=drop_rate)
        self.downsamplev=DownSample(in_dim=2*dim,out_dim=dim,drop_rate=drop_rate)
        

        self.fusion = ParameterFusion(dim=dim)


        


    def forward(self,v,a,ap,va_mask):
        
        ga = self.global_audio(a,va_mask).unsqueeze(1).expand(-1, v.shape[1], -1)
        gv = self.global_video(v,va_mask).unsqueeze(1).expand(-1, v.shape[1], -1)

        na = self.downsamplea(torch.cat([ga,a],dim=-1))
        nv = self.downsamplev(torch.cat([gv,v],dim=-1))


        av=self.fusion(nv,na,ap,va_mask)

        return av


class MultiScaleFusion(nn.Module):
    def __init__(self, dim,layers,  drop_rate=0.0):
        super(MultiScaleFusion, self).__init__()

        self.layers=layers

        self.rnn1 = nn.GRU( 2*dim,dim//2,num_layers=self.layers,batch_first=True,bidirectional=True)
        self.rnn2 = nn.GRU( 2*dim,dim//2,num_layers=self.layers,batch_first=True,bidirectional=True)
        self.rnn3 = nn.GRU( 2*dim,dim//2,num_layers=self.layers,batch_first=True,bidirectional=True)

        

        self.rnn123 = nn.GRU( 3*dim,dim,num_layers=1,batch_first=True,bidirectional=True)
        self.fusion=DownSample(in_dim =2*dim,out_dim=dim,drop_rate=drop_rate)


    def forward(self, f1,f2,f3,v_mask):
    
        length=torch.sum(v_mask,dim=1).cpu().int()




        f12=torch.cat([f1,f2],dim=-1)
        f12=pack_padded_sequence(f12,length,batch_first=True,enforce_sorted=False)
        f12,_ = self.rnn1(f12)
        f12=pad_packed_sequence(f12,batch_first=True)[0]

        f23=torch.cat([f2,f3],dim=-1)
        f23=pack_padded_sequence(f23,length,batch_first=True,enforce_sorted=False)
        f23,_ = self.rnn2(f23)
        f23=pad_packed_sequence(f23,batch_first=True)[0]

        f13=torch.cat([f1,f3],dim=-1)
        f13=pack_padded_sequence(f13,length,batch_first=True,enforce_sorted=False)
        f13,_ = self.rnn3(f13)
        f13=pad_packed_sequence(f13,batch_first=True)[0]

        f=torch.cat([f12,f23,f13],dim=-1)
        f=pack_padded_sequence(f,length,batch_first=True,enforce_sorted=False)
        f,_ = self.rnn123(f)
        f=pad_packed_sequence(f,batch_first=True)[0]
        f=self.fusion(f)


        return f

class MultiScaleFusion2(nn.Module):
    def __init__(self, dim,layers,  drop_rate=0.0):
        super(MultiScaleFusion2, self).__init__()

        self.layers=layers
        self.rnn = nn.GRU( 2*dim,dim//2,num_layers=self.layers,batch_first=True,bidirectional=True)
        # self.fusion=DownSample(in_dim =dim*3,out_dim=dim,drop_rate=drop_rate)


    def forward(self, f1,f2,v_mask):

        length=torch.sum(v_mask,dim=1).cpu().int()
        f=torch.cat((f1,f2),dim=-1)
        # f=self.dropout(f)
        f=pack_padded_sequence(f,length,batch_first=True,enforce_sorted=False)
        f,_ = self.rnn(f)
        f=pad_packed_sequence(f,batch_first=True)[0]
        # f=self.fusion(f)
        return f

