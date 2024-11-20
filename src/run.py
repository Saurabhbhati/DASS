# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong, Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--model_size", type=str, default='tiny224', help="the model size used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")
parser.add_argument('--pos_emb_type', type=str, default='learned',choices=['learned','none','sine'], help="the type of positional embedding used in the model")
parser.add_argument('--knowledge_distillation', type=ast.literal_eval, default='False', help="use knowledge distillation for the model")
parser.add_argument('--dist_loss_type',type=str,default='kldiv',choices=['l2c','kldiv','cosine','l2','bce'],help="the type of distillation loss used in the model")
parser.add_argument('--dist_loss_weight',type=float,default=0.5,help="the weight of distillation loss in the total loss")
parser.add_argument('--dist_temp',type=float,default=1.0,help="the temperature of distillation loss")
parser.add_argument('--dist_teacher_func',type=str,default='sigmoid',choices=['sigmoid','softmax'],help="the function applied to the teacher model output")
parser.add_argument('--kd_teach_type',type=str,default='ast',choices=['ast','DASS','self'],help="the type of teacher model used for knowledge distillation")
parser.add_argument('--kd_teach_dir',type=str,default='',help="the directory of the teacher model")
parser.add_argument('--kd_EMA_momentum',type=float,default=0.999,help="the momentum of the EMA teacher model")
parser.add_argument('--clf_input_emb',type=str,default='avg',choices=['first','last','mid','max','avg','cls','sum'],help="use input embedding for the classifier")
parser.add_argument('--ssm_ratio',type=float,default=1,help="ssm ratio")
parser.add_argument('--ssm_d_state',type=int,default=1,help="ssm d state")
parser.add_argument('--es_patience',type=int,default=5,help="early stopping patience, -1 means no early stopping")
parser.add_argument('--force_limit_batches_per_epoch',type=float,default=-1,help="effective num batches in epoch = force_limit_batches_per_epoch * len(train_loader), negative value means no limit, only used for full dataset")


args = parser.parse_args()
print(args)
# transformer based model

print('now train a audio spectrogram transformer model')

# 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
# dataset spectrogram mean and std, used to normalize the input
# norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
# target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
# # if add noise for data augmentation, only use for speech commands
# noise = {'audioset': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                'noise':args.noise}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model == 'ast':
    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size,pos_emb_type=args.pos_emb_type)
elif args.model == 'DASS':
    print('Using DASS model')
    audio_model = models.DASS(label_dim=args.n_class, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size)

print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

teacher_model = None
if args.knowledge_distillation:
    if args.kd_teach_type == 'ast':
        print('Using knowledge distillation from AST with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight) )
        teacher_model = models.ASTModel(label_dim=args.n_class, fstride=10, tstride=10, input_fdim=128,
                                        input_tdim=args.audio_length, imagenet_pretrain=True,
                                        audioset_pretrain=True, model_size='base384')

        if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
            import wget
            audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
            wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
        sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location='cpu')
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model)
        teacher_model.load_state_dict(sd, strict=True)
    elif args.kd_teach_type == 'DASS':
        print('Using knowledge distillation from DASS with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight))
        ## tiny mast model with 47.1 mAP
        if args.kd_teach_dir == '':
            print("Please specify the teacher model directory for knowledge distillation")
            exit()
        if 'small' in args.kd_teach_dir:
            model_size = 'small'
        elif 'medium' in args.kd_teach_dir:
            model_size = 'medium'
        teacher_model = models.DASS(label_dim=args.n_class, imagenet_pretrain=True,
                                        audioset_pretrain=False, model_size=model_size)
        sd = torch.load(args.kd_teach_dir, map_location='cpu')
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model).eval()
        teacher_model.load_state_dict(sd, strict=True)
    elif args.kd_teach_type == 'self':
        import copy
        print('Using knowledge distillation from self with loss type: {:s} loss temp {:.2f} loss weight {:.2f}'.format(args.dist_loss_type, args.dist_temp,args.dist_loss_weight))
        teacher_model = copy.deepcopy(audio_model)
        
        if not isinstance(teacher_model, torch.nn.DataParallel):
            teacher_model = torch.nn.DataParallel(teacher_model).eval()


print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model,teacher_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

