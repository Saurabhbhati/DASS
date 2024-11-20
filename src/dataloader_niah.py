# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong, Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetNIAHDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.niah_use_noise = self.audio_conf.get('niah_use_noise')
        self.niah_noise_type = self.audio_conf.get('niah_noise_type')
        self.niah_noise_snr = self.audio_conf.get('niah_noise_snr')
        self.niah_noise_max_length = self.audio_conf.get('niah_noise_max_length')
        print("NiAH conf: use_noise: {}, noise_type: {}, noise_snr: {} max_len: {}".format(self.niah_use_noise, self.niah_noise_type, self.niah_noise_snr, self.niah_noise_max_length))


    def __getitem__(self, index):
        datum = self.data[index]
        waveform_ori, sr = torchaudio.load(datum['wav'])
        waveform = waveform_ori - waveform_ori.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        downsampling_ratio = (waveform.shape[1] / fbank.shape[0])
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        if self.niah_use_noise == False:
            if index == 0:
                print("No noise: appending with zeros")
            fbank_noise = torch.zeros(self.niah_noise_max_length-target_length,128)
        else:
            if index == 0:
                print("Adding noise during evaluation")
            noise_len = int(downsampling_ratio * (self.niah_noise_max_length-target_length))+1
            if self.niah_noise_type == 'white':
                if index == 0:
                    print("Using white noise")
                noise_wav = torch.randn(1,noise_len)
            elif self.niah_noise_type == 'babble':
                if index == 0:
                    print("Using babble noise")
                noise_wav, noise_sr = torchaudio.load('../babble_all.wav')
                if noise_len > noise_wav.shape[1]:
                    ratio = int(np.ceil(noise_len/noise_wav.shape[1]))
                    noise_wav = torch.cat([noise_wav, noise_wav], dim=1)
                    noise_wav = noise_wav.repeat(1,ratio)
                noise_wav = noise_wav[:,:noise_len]
            else:
                Exception("Noise type not supported yet")
        
            clean_rms = torch.sqrt(torch.mean((waveform_ori ** 2),axis=-1))
            noise_rms = torch.sqrt(torch.mean((noise_wav ** 2),axis=-1))
            adjusted_noise_rms = clean_rms / (10**(self.niah_noise_snr/20))
            adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
            adjusted_noise_wav = adjusted_noise_wav - adjusted_noise_wav.mean()

            # compute the fbank features for the noise chunk by chunk
            chunk_size = 400*16000 # 400 seconds
            if adjusted_noise_wav.shape[1] < chunk_size:
                fbank_noise = torchaudio.compliance.kaldi.fbank(adjusted_noise_wav, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            else:
                fbank_noise = []
                for i in range(0, adjusted_noise_wav.shape[1], chunk_size):
                    fbank_chunk = torchaudio.compliance.kaldi.fbank(adjusted_noise_wav[:,i:i+chunk_size], htk_compat=True, sample_frequency=sr, use_energy=False,
                                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
                    fbank_noise.append(fbank_chunk)
                fbank_noise = torch.cat(fbank_noise, dim=0)
            if fbank_noise.shape[0] > self.niah_noise_max_length-target_length:
                fbank_noise = fbank_noise[:self.niah_noise_max_length-target_length,:]
            fbank_noise = (fbank_noise - self.norm_mean) / (self.norm_std * 2)

        return fbank, fbank_noise, label_indices

    def __len__(self):
        return len(self.data)
    
def collate_fn_niah(batch):
    fbank, fbank_noise, label_indices = zip(*batch)
    fbank = torch.stack(fbank, dim=0)
    fbank_noise = torch.stack(fbank_noise, dim=0)
    label_indices = torch.stack(label_indices, dim=0)
    return fbank, fbank_noise, label_indices