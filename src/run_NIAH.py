# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong, Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run_NIAH.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
import torch.nn as nn
from utilities import *
import pandas as pd

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-eval", type=str, default='/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/audioset/data/class_labels_indices.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='DASS', help="the model used")
parser.add_argument("--model_size", type=str, default='small', help="the model size used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("-ckpt","--checkpoint_dir",type=str, default="../pretrained_models/DASS_small.pth", help="directory to save checkpoints")

parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")


parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default='BCE', help="loss function", choices=["BCE", "CE"])

parser.add_argument('--pos_emb_type', type=str, default='learned',choices=['learned','none','sine'], help="the type of positional embedding used in the model")
parser.add_argument('--input_max_len', type=int, default=5000, help="the maximum length of the input sequence")
parser.add_argument('--audio_insert_tstep', type=float, default=0, help="the time step to insert audio data")
parser.add_argument('--clf_input_emb',type=str,default='all',choices=['first','last','mid','max','avg'],help="use input embedding for the classifier")
parser.add_argument('--niah_noise_snr',type=float,default=10,help="noise snr")
parser.add_argument('--niah_use_noise', help='disable noise during eval, use zeros, for debugging', type=ast.literal_eval, default='False')
parser.add_argument('--niah_noise_type', help='type of noise', type=str, default='white')

args = parser.parse_args()
print(args)
# transformer based model

print('now train a audio spectrogram transformer model')

val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

val_dataset = dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf)
val_loader = torch.utils.data.DataLoader(
    val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

import dataloader_niah
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False
                    ,'niah_use_noise':args.niah_use_noise,'niah_noise_type':args.niah_noise_type,'niah_noise_snr':args.niah_noise_snr,'niah_noise_max_length':args.input_max_len}
val_dataset1 = dataloader_niah.AudiosetNIAHDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf)
val_loader1 = torch.utils.data.DataLoader(
    val_dataset1,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    collate_fn=dataloader_niah.collate_fn_niah)

if args.model == 'ast':
    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size, pos_emb_type=args.pos_emb_type)
elif args.model == 'DASS':
    print('Using DASS model')
    audio_model = models.DASS(label_dim=args.n_class, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size)


if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model).eval()

audio_model = audio_model.cuda()
audio_model.load_state_dict(torch.load(args.checkpoint_dir), strict=True)

if args.model == 'ast' and args.input_max_len > 0:
    print("Overriding the positional embeddings in AST")
    f_dim, t_dim = audio_model.module.get_shape(10, 10, 128, args.input_max_len)
    new_pos_embed = models.get_sin_pos(f_dim*t_dim + 2, audio_model.module.v.pos_embed.shape[-1])
    new_pos_embed = nn.Parameter(new_pos_embed,requires_grad=False)
    audio_model.module.v.pos_embed = new_pos_embed


main_metrics = args.metrics
if args.loss == 'BCE':
    loss_fn = nn.BCEWithLogitsLoss()
elif args.loss == 'CE':
    loss_fn = nn.CrossEntropyLoss()

print(loss_fn)
args.loss_fn = loss_fn

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (fbank, fbank_noise, labels) in enumerate(val_loader1):

            if i % args.n_print_steps == 0:
                print('Validation: [{0}/{1}] Time {batch_time.avg:.5f}'.format(i, len(val_loader1),batch_time=batch_time))
            if args.input_max_len > 0:
                # if args.audio_insert_tstep >= 1.0:
                #     ind = args.input_max_len - fbank.shape[1]
                # else:
                #     ind = int(args.input_max_len * args.audio_insert_tstep)

                ind = int(args.input_max_len * args.audio_insert_tstep)
                if ind + args.audio_length > args.input_max_len:
                    ind = args.input_max_len - args.audio_insert_tstep
                
                audio_input = torch.cat((fbank_noise[:,:ind, :], fbank, fbank_noise[:,ind:, :]), dim=1)
            else:
                audio_input = fbank

            # compute output
            audio_output = audio_model(audio_input.to(device))
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)
    return stats, loss


stats, loss = validate(audio_model, val_loader, args, 0)

mAP = np.mean([stat['AP'] for stat in stats])
mAUC = np.mean([stat['auc'] for stat in stats])
acc = stats[0]['acc']
print('ckpt {} maxl {:.0f} niah_use_noise {} noisetype {} snr {:1f} tstep {:.4f} mAP: {:.4f}, mAUC: {:.4f}, acc: {:.4f}, loss: {:.4f}'.format(args.checkpoint_dir,args.input_max_len,
            args.niah_use_noise,args.niah_noise_type,args.niah_noise_snr,args.audio_insert_tstep, mAP, mAUC, acc, loss))

# save the results to a file 
file_path='./'
with open(os.path.join(file_path, 'niah_results_noise.txt'), 'a') as f:
    f.write('ckpt {} maxl {:.0f} niah_use_noise {} noisetype {} snr {:1f} tstep {:.4f} mAP: {:.4f}, mAUC: {:.4f}, acc: {:.4f}, loss: {:.4f}\n'.format(args.checkpoint_dir,args.input_max_len,
            args.niah_use_noise,args.niah_noise_type,args.niah_noise_snr,args.audio_insert_tstep, mAP, mAUC, acc, loss))

df = pd.DataFrame([[args.checkpoint_dir,args.niah_use_noise,args.niah_noise_type,args.niah_noise_snr, args.input_max_len, args.audio_insert_tstep, mAP, mAUC, acc, loss],],
                columns=['ckpt','niah_noise','noise_type','snr', 'maxl', 'tstep', 'mAP', 'mAUC', 'acc', 'loss'])

df_savepath = os.path.join(file_path, 'niah_results_noise.csv')
if os.path.exists(df_savepath):
    df2 = pd.read_csv(df_savepath)
    df = pd.concat([df2,df],ignore_index=True)

df.to_csv(df_savepath,index=False)

print("I am process %s, running on %s: finishing (%s)" % (os.getpid(), os.uname()[1], time.asctime()))