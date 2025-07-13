import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import CrossMultiHeadAttentionBlock, DownSample,Embedding, ParameterPredictor,VisualProjection,CQAttention,MultiScaleFusion,ConvBlockEncoder,HighLightLayer, \
    ConditionedPredictor,PositionalEmbedding, WeightedPool,FeatureEncoder, LocalAudioFusion,EventAudioFusion,GlobalAudioFusion,Encoder, \
            FusionLayerAV, SequentialQueryAttention, PositionalEncoding, VideoTextConcatenate,MultiHeadAttentionBlock
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from IPython import embed
import numpy as np


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler

def get_mean_feature(feature,mask):
    length = torch.sum(mask,dim=1)
    mean_feature=torch.zeros((feature.shape[0],feature.shape[2]),dtype=torch.float32).to(feature.device)
    for i in range(feature.shape[0]):
        mean_feature[i]=torch.mean(feature[i][:int(length[i])],dim=0)
    return mean_feature

class IMGNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(IMGNet, self).__init__()
        self.configs = configs
        self.word_embeddingv = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        self.word_embeddinga = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)


        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)
        self.audio_affine = VisualProjection(visual_dim=configs.audio_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)

        self.predictor_av = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.predictor_v = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.predictor_a = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)

        self.feature_encoderv=  Encoder(max_len=configs.max_pos_len,dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=2,
                                               drop_rate=configs.drop_rate)
        self.transformerv =  MultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        self.cross_transformerv =  CrossMultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        
        self.feature_encodera =  Encoder(max_len=configs.max_pos_len,dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=2,
                                              drop_rate=configs.drop_rate)
        self.transformera =  MultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        self.cross_transformera =  CrossMultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)

        self.cq_attentionv = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concatv = VideoTextConcatenate(dim=configs.dim)
        self.cq_attentiona = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concata = VideoTextConcatenate(dim=configs.dim)
        self.PPmodule= ParameterPredictor(dim=configs.dim)

        self.local_audio_fusion = LocalAudioFusion(dim=configs.dim,num_heads=configs.num_heads,drop_rate=configs.drop_rate)
        self.event_audio_fusion = EventAudioFusion(dim=configs.dim,num_heads=configs.num_heads,event_num=configs.event_num,slot_layers=configs.slot_layers,drop_rate=configs.drop_rate)
        self.global_audio_fusion = GlobalAudioFusion(dim=configs.dim,num_heads=configs.num_heads,drop_rate=configs.drop_rate)
        self.saliency_proj_v=nn.Linear(configs.dim,1)
        self.saliency_proj_a=nn.Linear(configs.dim,1)
        self.saliency_proj_av=nn.Linear(configs.dim,1)
        self.down_sample = MultiScaleFusion(dim=configs.dim,layers=configs.layers,drop_rate=configs.drop_rate)
        
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, va_mask, audio_features,q_mask,p_warm):
        #ENC
        video_features = self.video_affine(video_features)
        audio_features = self.audio_affine(audio_features)
        queryv_features = self.word_embeddingv(word_ids, char_ids)
        querya_features = self.word_embeddinga(word_ids, char_ids)

        #VT AT Interact
        video_features = self.feature_encoderv(video_features, mask=va_mask)
        queryv_features = self.feature_encoderv(queryv_features, mask=q_mask)

        audio_features = self.feature_encodera(audio_features, mask=va_mask)
        querya_features = self.feature_encodera(querya_features, mask=q_mask)


        video_features = self.transformerv(video_features,va_mask)
        audio_features = self.transformera(audio_features,va_mask)
        video_features,_ = self.cross_transformerv(video_features,queryv_features,va_mask.unsqueeze(1),q_mask.unsqueeze(1))
        audio_features,_ = self.cross_transformera(audio_features,querya_features,va_mask.unsqueeze(1),q_mask.unsqueeze(1))


        vt_features = self.cq_attentionv(video_features, queryv_features, va_mask, q_mask)
        vt_features = self.cq_concatv(vt_features, queryv_features, q_mask)

        at_features = self.cq_attentiona(audio_features, querya_features, va_mask, q_mask)
        at_features = self.cq_concata(at_features, querya_features, q_mask)


        v_score=self.saliency_proj_v(vt_features).squeeze(-1)
        a_score=self.saliency_proj_a(at_features).squeeze(-1)

        ap,o_ap = self.PPmodule(vt_features,at_features,va_mask,p_warm)

        fuse1 =  self.local_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse2 = self.event_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse3 =  self.global_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse = self.down_sample(fuse1,fuse2,fuse3,va_mask)
        av_score=self.saliency_proj_av(fuse).squeeze(-1)


        start_logits_av, end_logits_av = self.predictor_av(fuse, mask=va_mask)
        start_logits_v, end_logits_v = self.predictor_v(vt_features, mask=va_mask)
        start_logits_a, end_logits_a = self.predictor_a(at_features, mask=va_mask)
        return start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,o_ap.squeeze(1).squeeze(-1)
        
    def extract_index(self, start_logits, end_logits):
        return self.predictor_v.extract_index(start_logits=start_logits, end_logits=end_logits)
    

    def compute_ga_loss(self,v1,v2,p,th,tmp):
        label = torch.exp(v2/tmp) / (torch.exp(v1/tmp)+torch.exp(v2/tmp))
        boola = (label>th).int()
        label = label*boola
        label[label>(1-th)]=1

        return F.binary_cross_entropy(p,label),label,boola


    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor_v.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
    def compute_highlight_loss(self, scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss,loss_per_location
    
    def compute_distill_loss(self,student_start,student_end,teacher_start,teacher_end):
        return self.predictor_v.new_compute_distillation_loss(student_start=student_start,student_end=student_end,teacher_start=teacher_start,teacher_end=teacher_end)

    def compute_selective_distill_loss(self,student_start,student_end,teacher_start,teacher_end,matrix):
        return self.predictor_v.new_compute_selective_distillation_loss(student_start=student_start,student_end=student_end,teacher_start=teacher_start,teacher_end=teacher_end,matrix=matrix)

    def compute_score_loss(self,scores,pos,neg,margin=0.2):
        num_pairs = pos.shape[1]
        batch_indices = torch.arange(len(scores)).to(scores.device)
        pos_scores = torch.stack(
            [scores[batch_indices, pos[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [scores[batch_indices, neg[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(margin + neg_scores - pos_scores, min=0) \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        
        return torch.sum(loss_saliency,dim=-1)
    def compute_score(self,scores,pos,neg):
        num_pairs = pos.shape[1]
        batch_indices = torch.arange(len(scores)).to(scores.device)
        pos_scores = torch.stack(
            [scores[batch_indices, pos[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [scores[batch_indices, neg[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.mean(torch.clamp(pos_scores  -  neg_scores  , min=0),dim=1)
        
        return loss_saliency
    def compute_weighted_loss(self, start_logits, end_logits,
        start_logits_v, end_logits_v,start_logits_av, end_logits_av, start_labels, end_labels, grades, rho_th, s_v_th):
        rho, s_av, s_v = grades
        start_loss_a = nn.CrossEntropyLoss(reduction='none')(start_logits, start_labels) # not need softmax
        end_loss_a = nn.CrossEntropyLoss(reduction='none')(end_logits, end_labels)
        start_loss_v = nn.CrossEntropyLoss(reduction='none')(start_logits_v, start_labels)
        end_loss_v = nn.CrossEntropyLoss(reduction='none')(end_logits_v, end_labels)
        start_loss_av = nn.CrossEntropyLoss(reduction='none')(start_logits_av, start_labels)
        end_loss_av = nn.CrossEntropyLoss(reduction='none')(end_logits_av, end_labels)
        mask = torch.ones(rho.shape).type_as(rho)
        batch_size = rho.shape[0]
        for i in range(batch_size):
            if rho[i] < rho_th and s_v[i] > s_v_th:
                mask[i] = 0.0
        return ((start_loss_a + end_loss_a)*mask + (start_loss_v+end_loss_v)+ (start_loss_av+end_loss_av)).mean()


class IMGNet4VLM(nn.Module):
    def __init__(self, configs, word_vectors):
        super(IMGNet4VLM, self).__init__()
        self.configs = configs
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)
        self.audio_affine = VisualProjection(visual_dim=configs.audio_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)

        self.text_affinev = VisualProjection(visual_dim=configs.text_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)
        self.text_affinea = VisualProjection(visual_dim=configs.text_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)

        self.predictor_av = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.predictor_v = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.predictor_a = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)

        self.feature_encoderv=  Encoder(max_len=configs.max_pos_len,dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=2,
                                               drop_rate=configs.drop_rate)
        self.transformerv =  MultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        self.cross_transformerv =  CrossMultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        
        self.feature_encodera =  Encoder(max_len=configs.max_pos_len,dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=2,
                                              drop_rate=configs.drop_rate)
        self.transformera =  MultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)
        self.cross_transformera =  CrossMultiHeadAttentionBlock(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate)

        self.cq_attentionv = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concatv = VideoTextConcatenate(dim=configs.dim)
        self.cq_attentiona = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concata = VideoTextConcatenate(dim=configs.dim)
        self.PPmodule= ParameterPredictor(dim=configs.dim)

        self.local_audio_fusion = LocalAudioFusion(dim=configs.dim,num_heads=configs.num_heads,drop_rate=configs.drop_rate)
        self.event_audio_fusion = EventAudioFusion(dim=configs.dim,num_heads=configs.num_heads,event_num=configs.event_num,slot_layers=configs.slot_layers,drop_rate=configs.drop_rate)
        self.global_audio_fusion = GlobalAudioFusion(dim=configs.dim,num_heads=configs.num_heads,drop_rate=configs.drop_rate)
        self.saliency_proj_v=nn.Linear(configs.dim,1)
        self.saliency_proj_a=nn.Linear(configs.dim,1)
        self.saliency_proj_av=nn.Linear(configs.dim,1)
        self.down_sample = MultiScaleFusion(dim=configs.dim,layers=configs.layers,drop_rate=configs.drop_rate)
        
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, video_features, va_mask, audio_features,text_features,q_mask,p_warm):
        #ENC
        video_features = self.video_affine(video_features)
        audio_features = self.audio_affine(audio_features)
        queryv_features = self.text_affinev(text_features)
        querya_features = self.text_affinea(text_features)

        #VT AT Interact
        video_features = self.feature_encoderv(video_features, mask=va_mask)
        queryv_features = self.feature_encoderv(queryv_features, mask=q_mask)

        audio_features = self.feature_encodera(audio_features, mask=va_mask)
        querya_features = self.feature_encodera(querya_features, mask=q_mask)


        video_features = self.transformerv(video_features,va_mask)
        audio_features = self.transformera(audio_features,va_mask)
        video_features,_ = self.cross_transformerv(video_features,queryv_features,va_mask.unsqueeze(1),q_mask.unsqueeze(1))
        audio_features,_ = self.cross_transformera(audio_features,querya_features,va_mask.unsqueeze(1),q_mask.unsqueeze(1))


        vt_features = self.cq_attentionv(video_features, queryv_features, va_mask, q_mask)
        vt_features = self.cq_concatv(vt_features, queryv_features, q_mask)

        at_features = self.cq_attentiona(audio_features, querya_features, va_mask, q_mask)
        at_features = self.cq_concata(at_features, querya_features, q_mask)


        v_score=self.saliency_proj_v(vt_features).squeeze(-1)
        a_score=self.saliency_proj_a(at_features).squeeze(-1)

        ap,o_ap = self.PPmodule(vt_features,at_features,va_mask,p_warm)

        fuse1 =  self.local_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse2 = self.event_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse3 =  self.global_audio_fusion(vt_features,at_features,ap,va_mask)
        fuse = self.down_sample(fuse1,fuse2,fuse3,va_mask)
        av_score=self.saliency_proj_av(fuse).squeeze(-1)


        start_logits_av, end_logits_av = self.predictor_av(fuse, mask=va_mask)
        start_logits_v, end_logits_v = self.predictor_v(vt_features, mask=va_mask)
        start_logits_a, end_logits_a = self.predictor_a(at_features, mask=va_mask)
        return start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,o_ap.squeeze(1).squeeze(-1)
        
    def extract_index(self, start_logits, end_logits):
        return self.predictor_v.extract_index(start_logits=start_logits, end_logits=end_logits)
    

    def compute_ga_loss(self,v1,v2,p,th,tmp):
        label = torch.exp(v2/tmp) / (torch.exp(v1/tmp)+torch.exp(v2/tmp))
        boola = (label>th).int()
        label = label*boola
        label[label>(1-th)]=1

        return F.binary_cross_entropy(p,label),label,boola


    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor_v.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
    def compute_highlight_loss(self, scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss,loss_per_location
    
    def compute_distill_loss(self,student_start,student_end,teacher_start,teacher_end):
        return self.predictor_v.new_compute_distillation_loss(student_start=student_start,student_end=student_end,teacher_start=teacher_start,teacher_end=teacher_end)

    def compute_selective_distill_loss(self,student_start,student_end,teacher_start,teacher_end,matrix):
        return self.predictor_v.new_compute_selective_distillation_loss(student_start=student_start,student_end=student_end,teacher_start=teacher_start,teacher_end=teacher_end,matrix=matrix)

    def compute_score_loss(self,scores,pos,neg,margin=0.2):
        num_pairs = pos.shape[1]
        batch_indices = torch.arange(len(scores)).to(scores.device)
        pos_scores = torch.stack(
            [scores[batch_indices, pos[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [scores[batch_indices, neg[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(margin + neg_scores - pos_scores, min=0) \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        
        return torch.sum(loss_saliency,dim=-1)
    def compute_score(self,scores,pos,neg):
        num_pairs = pos.shape[1]
        batch_indices = torch.arange(len(scores)).to(scores.device)
        pos_scores = torch.stack(
            [scores[batch_indices, pos[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [scores[batch_indices, neg[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.mean(torch.clamp(pos_scores  -  neg_scores  , min=0),dim=1)
        
        return loss_saliency
    def compute_weighted_loss(self, start_logits, end_logits,
        start_logits_v, end_logits_v,start_logits_av, end_logits_av, start_labels, end_labels, grades, rho_th, s_v_th):
        rho, s_av, s_v = grades
        start_loss_a = nn.CrossEntropyLoss(reduction='none')(start_logits, start_labels) # not need softmax
        end_loss_a = nn.CrossEntropyLoss(reduction='none')(end_logits, end_labels)
        start_loss_v = nn.CrossEntropyLoss(reduction='none')(start_logits_v, start_labels)
        end_loss_v = nn.CrossEntropyLoss(reduction='none')(end_logits_v, end_labels)
        start_loss_av = nn.CrossEntropyLoss(reduction='none')(start_logits_av, start_labels)
        end_loss_av = nn.CrossEntropyLoss(reduction='none')(end_logits_av, end_labels)
        mask = torch.ones(rho.shape).type_as(rho)
        batch_size = rho.shape[0]
        for i in range(batch_size):
            if rho[i] < rho_th and s_v[i] > s_v_th:
                mask[i] = 0.0
        return ((start_loss_a + end_loss_a)*mask + (start_loss_v+end_loss_v)+ (start_loss_av+end_loss_av)).mean()