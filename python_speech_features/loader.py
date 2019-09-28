"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
import librosa
import python_speech_features as features
import scipy.io.wavfile as wav
from scipy import signal
import numpy as np
from python_speech_features import logfbank
from torch.utils.data import Dataset, DataLoader

### test 용 library ###
import matplotlib.pyplot as plt
from librosa import display

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

target_dict = dict()

# Make dictionary as {'wav_001': '192 755 662 192 678 476 662 408 690 2 125 610 662 220 640 125 662 179 192 661 123 662'}
def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

# tmp_file_path = "./sample_dataset/train/train_data/wav_001.wav"
# sig, sample_rate = librosa.core.load(tmp_file_path, SAMPLE_RATE)
# lib_mfcc = librosa.feature.mfcc(sig, sr = sample_rate, n_mfcc=40, n_fft = N_FFT, hop_length = 128)
# lib_melspec = librosa.feature.melspectrogram(sig, n_mels = 40, n_fft = 512, hop_length = 128)

# mfcc feature: (513, 13)
# librosa: (40, 644)

tmp_filepath = "./sample_dataset/train/train_data/wav_001.wav"

########### spectogram에 log 씌움 [257, 257]############
def log_specgram(filepath, SAMPLE_RATE, N_FFT, window_size=30, step_size=10, eps=1e-10):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()
    # nperseg: Length of each segment
    # noverlap: Number of points to overlap between segments
    nperseg = int(round(window_size * SAMPLE_RATE / 1e3))
    noverlap = int(round(step_size * SAMPLE_RATE / 1e3))
    freqs, times, spec = signal.spectrogram(sig, fs=SAMPLE_RATE,
                                            window='hann', nperseg=nperseg,
                                            noverlap=noverlap, nfft=N_FFT, detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

log_specgram(tmp_filepath, SAMPLE_RATE, N_FFT)[2].shape

# log_specgram normalize하고 filterbank 거침 [411, 26]
def get_features_from_spectrogram_with_filterbank(filepath, sample_rate, N_FFT, window_size=30, step_size=10, eps=1e-10):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()
    # nperseg: Length of each segment
    # noverlap: Number of points to overlap between segments
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(sig, fs=sample_rate, window = 'hann',
                                            nperseg=nperseg, noverlap=noverlap, nfft=N_FFT, detrend=False)
    mean = np.mean(spec, axis=0)
    std = np.std(spec, axis=0)
    spec = (spec - mean) / std
    fbank_feat = logfbank(spec.T.astype(np.float32)+eps, sample_rate, winlen=0.030,winstep=0.01,
      nfilt=26,nfft=512,lowfreq=0,highfreq=sample_rate/4,preemph=0.97)
    return freqs, times, fbank_feat

########### scipy 이용해서 2D plot 그리기 #############
freqs, times, spectrogram = get_features_from_spectrogram_with_filterbank(tmp_filepath, SAMPLE_RATE, N_FFT)
fig = plt.figure(figsize=(14, 4))
ax2 = fig.add_subplot(111)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

########### librosa 의 melspetrogram으로 2D plot 그리기 (40, 516)#############
mel_spectrogram = get_feature_from_librosa(tmp_filepath)
plt.figure(figsize=(10, 4))
display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# mfcc feature ([514, 13])
def get_mfcc_feature(filepath):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()
    signal = torch.FloatTensor(sig)
    feat = features.mfcc(signal, samplerate = SAMPLE_RATE, winlen=0.030, winstep=0.01, numcep=13,
                 nfilt=40, nfft=N_FFT, lowfreq=0, highfreq=None, preemph=0.97,
                ceplifter=22, appendEnergy=True)
    return torch.FloatTensor(feat)

# mfcc 값에 delta, acc 추가 ([514, 39])
def create_mfcc(filepath):
    (rate, sample) = wav.read(filepath)
    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
    preemph=0.97, appendEnergy=True)
    d_mfcc = features.delta(mfcc, 2)
    a_mfcc = features.delta(d_mfcc, 2)
    out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)
    return out, out.shape

# mel_spectrum(output shape: (512, 257))
def get_mel_spectrum_feature(filepath):
    feat = get_spectrogram_feature(filepath)
    mel_feat = 2595 * torch.log10(1+feat/700)
    return mel_feat

# Make STFT feature (output shape: (512, 257))
def get_spectrogram_feature(filepath):
    # sig.shape: (82400, 1)
    (rate, width, sig) = wavio.readwav(filepath)

    # len(sig): 82400
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)
    return feat

# Make spectrogram feature(output shape: (40, 644))
def get_feature_from_librosa(filepath):
    global first
    global sig
    global sample_rate

    sample_rate = 16000
    #hop_length = 128
    hop_length = int(0.01*sample_rate)
    sig, sample_rate = librosa.core.load(filepath, sample_rate)
    assert sample_rate == 16000, '%s sample rate must be 16000 but sample-rate is %d'% (filepath, sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(sig, n_mels = 40, n_fft = 512, hop_length = hop_length)

    return mel_spectrogram

## plot 2D spectrogram
# import matplotlib.pyplot as plt
# from librosa import display
# import numpy as np
#
# plt.figure(figsize = (10, 4))
# display.specshow(librosa.power_to_db(tmp_librosa_feat, ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()

# [192 755 662 192 678 476]
def get_script(filepath, bos_id, eos_id):
    '''
    :return: array as [1307, 192, 755, 662, 192, 678, 476, 1308] with target labels
    '''
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_mfcc_feature(self.wav_paths[idx])
        #feat = get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

