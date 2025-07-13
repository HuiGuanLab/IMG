import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model.IMGNet import IMGNet,IMGNet4VLM, build_optimizer_and_scheduler
from util.data_util import load_i3d_video_features,load_internvideo_features,load_video_clip_sf_features, save_json, load_json, load_audio_features,fix_json
from util.data_gen import gen_or_load_dataset
from util.data_gen_iv2 import gen_or_load_dataset_iv2
from util.data_loader import get_train_loader, get_test_loader
from util.runner_utils import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, \
    get_last_checkpoint, grader
import numpy as np
import faulthandler
from IPython import embed
from torch.utils.tensorboard import SummaryWriter
from warm_up import ExpUp


faulthandler.enable()
parser = argparse.ArgumentParser()

parser.add_argument('--lamb', type=float, default=1, help='the lambda in loss function')
parser.add_argument('--test_path', type=str, default=None, help='the path of the trained model when testing')
parser.add_argument("--save_predictions",type=str,default=None,help='the path to save predictions')

# data parameters
parser.add_argument('--save_dir', type=str, default='datasets', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charadesAM', help='target task, [charades|activitynet|charadesAM]')
# predict clips are cut by visual
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed for Visual, 128 for c3d, 128*3 for VGG')
parser.add_argument('--max_pos_len_a', type=int, default=128*3, help='maximal position sequence length allowed for Audio, 128 for VGGish, 128*3 for PANN')
# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=2048, help="video feature input dimension, 1024 for c3d, 4096 for VGG")
parser.add_argument("--audio_feature_dim", type=int, default=128, help="audio feature input dimension, 128 for VGGish, 2048 for PANN")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
# training/evaluation parameters
parser.add_argument('--visual_features', type=str, default='i3d', help='For visual features [i3d|clip+sf|iv2]')

parser.add_argument("--gpu_idx", type=str, default="3", help="GPU index")
parser.add_argument("--seed", type=int, default=None, help="random seed set [None|number] ")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0005, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='IMG', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
parser.add_argument('--constractive', type=bool, default=False, help='consloss')
parser.add_argument('--cons_lamb', type=float, default=1, help='consloss')
parser.add_argument('--event_num', type=int, default=3, help='event num')
parser.add_argument('--warm_epoch', type=int, default=20, help='loc loss weight')
parser.add_argument('--layers', type=int, default=3, help='rnn layers')
parser.add_argument('--our_loss_th', type=float, default=0.2, help='thershold')
parser.add_argument('--tmp', type=float, default=3, help='temperture')
parser.add_argument('--ia_lamb', type=int, default=5, help='Important-Aware lamb')
parser.add_argument('--sa_lamb', type=float, default=0.5, help='saliency lamb')
parser.add_argument('--kl_lamb', type=float, default=10, help='KD lamb')
parser.add_argument('--slot_layers', type=int, default=3, help='slotAtt layers')
parser.add_argument('--max_n', type=int, default=2, help='saliency num')
configs = parser.parse_args()

if configs.task == 'charades' or configs.task == 'charadesAM':
    # I3D visual features
    configs.tmp = 3
    configs.max_pos_len=128
    configs.our_loss_th = 0.2
    audio_features = load_audio_features(os.path.join('data', 'features', 'charades', "audio"), None, mode='PANN')
    if configs.visual_features=="i3d":
        configs.video_feature_dim = 1024
        visual_feature_path="data/features/charades/i3d_video"
        visual_features,audio_features = load_i3d_video_features(visual_feature_path, configs.max_pos_len,audio_features)
        new_json,old_json = fix_json(os.path.join('data', 'features', 'charades',"feature_shapes_i3d_v.json"),os.path.join('data', 'features', 'charades',"feature_shapes_a.json"))
    elif configs.visual_features=="iv2":
        configs.max_pos_len=64
        configs.video_feature_dim = 768
        configs.text_feature_dim = 4096
        visual_feature_path="data/features/charades/iv2_features"
        visual_features,audio_features,text_features = load_internvideo_features(visual_feature_path, configs.max_pos_len,audio_features)
        new_json,old_json = fix_json(os.path.join('data', 'features', 'charades',"feature_shapes_iv2_v.json"),os.path.join('data', 'features', 'charades',"feature_shapes_a.json"))
    elif configs.visual_features=="clip+sf":
        configs.video_feature_dim = 2816
        configs.text_feature_dim = 512
        visual_feature_path="data/features/charades/clip_features"
        visual_features,audio_features,text_features = load_video_clip_sf_features(visual_feature_path, configs.max_pos_len,audio_features)
        new_json,old_json = fix_json(os.path.join('data', 'features', 'charades',"feature_shapes_clip_v.json"),os.path.join('data', 'features', 'charades',"feature_shapes_a.json"))
        
        
    configs.max_pos_len_a = int(configs.max_pos_len*2)
    configs.audio_feature_dim = 2048


else:
    # C3D visual features
    configs.video_feature_dim = 1024
    configs.tmp = 2
    configs.our_loss_th=0.1
    audio_features = load_audio_features(os.path.join('data', 'features', configs.task, "audio/VGGish.pickle"), None ,mode='VGGish')
    if configs.visual_features=="i3d":
        configs.video_feature_dim = 1024
        visual_feature_path="data/features/activitynet/i3d_video"
        visual_features,audio_features = load_i3d_video_features(visual_feature_path, configs.max_pos_len,audio_features)
        new_json,old_json = fix_json(os.path.join('data', 'features', 'activitynet',"feature_shapes_i3d_v.json"),os.path.join('data', 'features', 'activitynet',"feature_shapes_a.json"))
    # elif configs.visual_features=="iv2":
    #     configs.video_feature_dim = 3200
    #     configs.text_feature_dim = 4096
    #     visual_feature_path="data/features/activitynet/iv2_features"
    #     visual_features,audio_features = load_internvideo_feature(visual_feature_path, configs.max_pos_len,audio_features)
    #     new_json,old_json = fix_json(os.path.join('data', 'features', 'activitynet',"feature_shapes_iv2_v.json"),os.path.join('data', 'features', 'charades',"feature_shapes_a.json"))
    ##no slowfast
    # elif configs.visual_features=="clip+sf":
    #     configs.video_feature_dim = 512
    #     visual_feature_path="data/features/charades/charades_clip_unimodal"
    #     visual_features,audio_features = load_video_clip_sf_features("/media/disk2/lja/charades_clip_features/charades_clip_unimodal", configs.max_pos_len,audio_features)
    #     new_json,old_json = fix_json(os.path.join('data', 'features', 'activitynet',"feature_shapes_clip_v.json"),os.path.join('data', 'features', 'charades',"feature_shapes_a.json")))
    # VGGish audio features
    configs.max_pos_len_a = int(128/1.8)
    configs.audio_feature_dim = 128
   



set_th_config(configs.seed)

exp_up=ExpUp(num_loops=configs.warm_epoch)
p_up=ExpUp(num_loops=10)

if configs.visual_features=="iv2":
    dataset = gen_or_load_dataset_iv2(configs,new_json,old_json)
else:
    dataset = gen_or_load_dataset(configs,new_json,old_json)
configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']
if configs.visual_features == "i3d":
    text_features=None
train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, audio_features=audio_features,text_features=text_features, configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset=dataset['val_set'], video_features=visual_features, audio_features=audio_features,text_features=text_features, configs=configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, audio_features=audio_features,text_features=text_features,configs=configs)
configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)
num_val_batches = 0 if val_loader is None else len(val_loader)
num_test_batches = len(test_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task,configs.visual_features,
                                                     'v_'+str(configs.max_pos_len), 'a_'+str(configs.max_pos_len_a)]))
tb_writer=SummaryWriter(log_dir="runs/"+home_dir)
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")


# train and test
if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    if text_features==None:
        model = IMGNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    else:
        model = IMGNet4VLM(configs=configs, word_vectors=dataset['word_vector']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs) # warm_up and weight_reduce are related to num_train_steps
    # start training
    best_r1i7 = -1.0
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    score_writera = open(os.path.join(model_dir, "eval_resultsa.txt"), mode="w", encoding="utf-8")
    score_writerv = open(os.path.join(model_dir, "eval_resultsv.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    global_step = 0


    for epoch in range(configs.epochs):
        model.train()
        sum_ga_loss=0
        sum_av_loss=0
        sum_a_loss=0
        sum_v_loss=0
        sum_kl_loss=0
        sum_valid=0
        warm = exp_up.get_value(epoch)
        p_warm = p_up.get_value(epoch)
        for data in tqdm(train_loader, total=num_train_batches, desc='Epoch %3d / %3d' % (epoch + 1, configs.epochs)):
            # len(train_loader) batches
            global_step += 1
            
            
            _ , vfeats, vfeat_lens, afeats,tfeats, tfeat_lens,  word_ids, char_ids, s_labels, e_labels,h_labels,pos,neg = data
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            afeats= afeats.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels,h_labels = s_labels.to(device), e_labels.to(device),h_labels.to(device)
            pos,neg=pos.to(device),neg.to(device)

            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute logits
            if tfeats!=None:
                tfeats, tfeat_lens = tfeats.to(device), tfeat_lens.to(device)
                text_mask = convert_length_to_mask(tfeat_lens).to(device)
                start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,param_a = \
                    model(vfeats, video_mask, afeats,tfeats, text_mask,p_warm)
            else:
                start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,param_a = \
                    model(word_ids, char_ids, vfeats, video_mask, afeats, query_mask,p_warm)

            loc_loss_av = model.compute_loss(start_logits_av, end_logits_av, s_labels, e_labels)
            loc_loss_v = model.compute_loss(start_logits_v, end_logits_v, s_labels, e_labels)
            loc_loss_a = model.compute_loss(start_logits_a, end_logits_a, s_labels, e_labels)


            vs_loss = model.compute_score_loss(v_score,pos,neg)
            as_loss = model.compute_score_loss(a_score,pos,neg)
            avs_loss = model.compute_score_loss(av_score,pos,neg)

            kl_loss1 = model.compute_distill_loss(start_logits_a, end_logits_a,start_logits_av, end_logits_av)
            kl_loss2 = model.compute_distill_loss(start_logits_v, end_logits_v,start_logits_av, end_logits_av)

            sum_kl_loss+=kl_loss1.mean().item()+kl_loss2.mean().item()



            sum_v_loss+=loc_loss_v.mean().item()
            sum_a_loss+=loc_loss_a.mean().item()
            sum_av_loss+=loc_loss_av.mean().item()

   
            ga_loss,labela,boola = model.compute_ga_loss(loc_loss_a.detach(),loc_loss_v.detach(),param_a,configs.our_loss_th,configs.tmp)

            valid=torch.sum(boola)

            sum_valid=sum_valid+labela.shape[0]-valid

            total_loss =  loc_loss_av.mean()+ loc_loss_v.mean() + torch.sum(boola*loc_loss_a)/(valid+1e-6) # location loss
            total_loss += configs.sa_lamb * (torch.sum(boola*as_loss)+avs_loss.sum()+vs_loss.sum())  # saliency loss
            total_loss += configs.kl_lamb * warm *(torch.sum(boola*kl_loss1)/(valid+1e-6)+kl_loss2.mean()) # kl loss
            total_loss += configs.ia_lamb * ga_loss             # Importantn-Aware Module loss 
            optimizer.zero_grad()
            total_loss.backward()
            sum_ga_loss=sum_ga_loss+ga_loss.item()
            
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)
            optimizer.step()
            scheduler.step()
            
            if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                if global_step % num_train_batches == 0:
                    tb_writer.add_scalar('av loss',sum_av_loss/num_train_batches,epoch)
                    tb_writer.add_scalar('v loss',sum_v_loss/num_train_batches,epoch)
                    tb_writer.add_scalar('a loss',sum_a_loss/num_train_batches,epoch)
                    tb_writer.add_scalar('kl loss',sum_kl_loss/num_train_batches,epoch)
                    tb_writer.add_scalar('ga loss',sum_ga_loss/num_train_batches,epoch)
                else:
                    eval_period+=num_train_batches
                model.eval()
                r1i3, r1i5, r1i7, mi, score_str,score_stra ,score_strv = eval_test(model=model, data_loader=test_loader, device=device,
                                                            mode='test', epoch=epoch + 1, global_step=global_step,configs=configs)
                print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                    epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)
                score_writer.write(score_str)
                score_writer.flush()
                score_writera.write(score_stra)
                score_writera.flush()
                score_writerv.write(score_strv)
                score_writerv.flush()
                if r1i7 >= best_r1i7:
                    best_r1i7 = r1i7
                    torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}.t7'.format(configs.model_name,
                                                                                            global_step)))
                    filter_checkpoints(model_dir, suffix='t7', max_to_keep=3)
                model.train()          

    score_writer.close()
    score_writera.close()
    score_writerv.close()
elif configs.mode.lower() == 'test':
    # if configs.test_path != None:
    #     model_dir = configs.test_path
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    model = IMGNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    filename=get_last_checkpoint(model_dir)
    model.load_state_dict(torch.load(filename,map_location=device),strict=True)
    model.eval()

    r1i3, r1i5, r1i7, mi, *_ = eval_test(model=model, data_loader=test_loader, device=device, mode='test',configs=configs)
    print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)
