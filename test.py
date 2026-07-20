import torch, torchaudio
import os, glob
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import random
import pickle
import pandas as pd
import csv
from torch.nn import functional as F
from torchaudio.functional import resample
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                          
from trident import Encoder
exist_flag = False
model_src = './checkpoints/Trident'

hparams_src = 'hyperparams_new_model.yaml'
savedir = model_src
enc = Encoder.from_hparams(source=model_src, hparams_file=hparams_src,
                                                    savedir=savedir, freeze_params=True, run_opts={"device": device, "data_parallel_backend": True})

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clean_dir', type=str, required=True, help='Path to clean dataset directory')
parser.add_argument('--convert_dir', type=str, required=True, help='Path to converted dataset directory')
args = parser.parse_args()

ori_path = args.clean_dir
test_path = args.convert_dir

vc_name = os.path.basename(test_path.rstrip('/'))

model_name = 'Nformat'

print(f"Clean dir: {ori_path}")
print(f"Convert dir: {test_path}")
print(f"VC method: {vc_name}")

def audio_loader(tar_wav, duration):
    num_frames = 16000 * duration
    if isinstance(tar_wav, list):
        sigs = []
        for wav in tar_wav:
            sig, fs = torchaudio.load(wav)
            sig = resample(sig, fs, 16000)
            sig = sig.transpose(0, 1).squeeze(1)
            sigs.append(sig)
        sig = torch.hstack(sigs)[0:num_frames]

        return sig
    else:
        if tar_wav != 'None':
            sig, fs = torchaudio.load(tar_wav)
            sig = resample(sig, fs, 16000)

            if sig.size(-1) < num_frames:
                sig = F.pad(sig, (0, num_frames-sig.size(-1)), "constant", 0)

            else:
                start = random.randint(0, sig.size(-1)-num_frames)
                sig = sig[:, start:start+num_frames]

            tar_sig = sig.transpose(0, 1).squeeze(1)

        else:
            tar_sig = 0 * torch.randn(num_frames)

        return tar_sig

def cosine_score(emb1,emb2):
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = similarity(emb1, emb2)
    return score

def random_select(list, num):
    list.sort()
    random.shuffle(list)
    return list[0:num]

def stop_condition(positive, negative, threshold, flag):
    ar = torch.tensor(positive)
    fnr = torch.sum(ar<threshold) / len(positive)
    ar = torch.tensor(negative)
    fpr = torch.sum(ar>threshold) / len(negative)
    if flag == 'Done':
        print('Equal Error Rate:%.5f'%fpr)
        return 0

    if abs(fnr - fpr) < 0.0001:
        factor = 0
        print('Equal Error Rate:%.5f'%fpr)

    else:
        if fpr > fnr:
            factor = 1
        else:
            factor = -1

    return factor

def eer_calculator(positive, negative):
    threshold = 0.2
    factor = stop_condition(positive, negative, threshold, 'Do')
    tmp = factor
    while factor != 0:
        if tmp == -factor:
            stop_condition(positive, negative, threshold, 'Done')
            break
        threshold = threshold + factor * 0.00005
        tmp = factor
        factor = stop_condition(positive, negative, threshold, 'Do')
        
    return threshold


def vp_revelio(wav):
    duration = 20
    
    wav = audio_loader(wav, duration)

    revelio_emb = enc.encode_batch(wavs=wav, tarwavs=torch.zeros_like(wav))

    remb = (revelio_emb / torch.linalg.norm(revelio_emb, dim=-1).unsqueeze(dim=0)).squeeze(dim=0).cpu()
    
    return remb 

def vp_cal(wavs):
    if isinstance(wavs, list) == False:
        for wav in wavs:
            return [vp_revelio(wav)]
    else:
        vps = []
        for wav in wavs:
            vps.append(vp_revelio(wav))
        return vps
        

def batch_cos(vp1, vp2s):
    scores = []
    for vp2 in vp2s:
        cos_score = cosine_score(vp1, vp2)
        if not torch.isnan(cos_score):
            scores.append(cos_score)
    return scores


def dev_test(testset, ori_path, test_path, vp_dict):
    positive = []
    negative = []
    for vc_wav in tqdm(testset):
        try:
            pair = vc_wav.split('/')[-2]
            atk_id, tar_id = pair.split('-')
            vc_wavs = random_select(glob.glob(test_path + '/' + pair + '/*'), 5)
            restored_vp = vp_revelio(vc_wavs)

            top_id, top_value, sampled_scores = return_top1_and_random_avg(vp_dict, restored_vp, exclude_id=atk_id)
            for sampled_score in sampled_scores:
                negative.extend(sampled_score)
            if atk_id != top_id:
                vp = vp_dict[atk_id]
                positive.extend(batch_cos(restored_vp, vp))
            else:
                positive.extend(top_value)
        except:
            continue
            
    return positive, negative

def dev_test_clean(testset, ori_path, test_path, vp_dict):
    positive = []
    negative = []
    for vc_wav in tqdm(testset):
        try:
            clean_id = vc_wav.split('/')[-2]
            atk_wavs = random_select(glob.glob(ori_path + '/' + clean_id + '/*'), 5)
            restored_vp = vp_revelio(atk_wavs)

            top_id, top_value, sampled_scores = return_top1_and_random_avg(vp_dict, restored_vp, exclude_id=clean_id)
            for sampled_score in sampled_scores:
                negative.extend(sampled_score)
            if clean_id != top_id:
                vp = vp_dict[clean_id]
                positive.extend(batch_cos(restored_vp, vp))
            else:
                positive.extend(top_value)
        except:
            continue
            
    return positive, negative

def prase_testfile(test_path):
    testset = []
    pairs = os.listdir(test_path)
    for pair in pairs:
        utts = glob.glob(test_path + '/' + pair + '/*')
        testset.extend(random_select(utts, 2))
    return testset

def sv_test(test_path, ori_path):
    print('#################################')
    print('Starting SV test')
    vp_dict = osi_vp_cal(test_path, ori_path)
    testset = prase_testfile(test_path)

    if vc_name != 'Clean':
        positive, negative = dev_test(testset, ori_path, test_path, vp_dict)
        threshold = eer_calculator(positive, negative)
    else:
        positive, negative = dev_test_clean(testset, ori_path, test_path, vp_dict)
        threshold = eer_calculator(positive, negative)
    return threshold
    
def osi_vp_cal(test_path, ori_path):
    spks = []
    vps = []
    spks = os.listdir(ori_path)
    print('speaker_num:%d'%len(spks))
    for spk in tqdm(spks):
        utts = glob.glob(ori_path + '/' + spk + '/*')
        utts = random_select(utts, 3)
        vp = vp_cal(utts)
        vp = torch.vstack(vp).squeeze(dim=1)
        vp = torch.mean(vp, dim=0).unsqueeze(dim=0)
        vps.append(vp)
    vp_dict = dict(zip(spks, vps))
    return vp_dict

def return_top1_and_random_avg(vp_dict, vp, exclude_id=None, n_samples=100):
    ids = list(vp_dict.keys())
    vps = [vp_dict[k] for k in ids]
    
    cos_scores = batch_cos(vp, vps)

    top_index = cos_scores.index(max(cos_scores))
    top1_id, top1_score = ids[top_index], max(cos_scores)

    exclude_id = exclude_id if exclude_id is not None else top1_id

    candidate_ids = [id_ for id_ in ids if id_ != exclude_id]

    sampled_ids = random.sample(candidate_ids, min(n_samples, len(candidate_ids)))
    
    sampled_scores = [cos_scores[ids.index(id_)] for id_ in sampled_ids]
    
    return top1_id, top1_score, sampled_scores


def osi_dev(vp_dict, testset, ori_path, top_xs, test_path):
    ids = list(vp_dict.keys())
    ranks = []
    for vc_wav in tqdm(testset):
        try:
            pair = vc_wav.split('/')[-2]
            atk_id, tar_id = pair.split('-')
            vc_wavs = random_select(glob.glob(test_path + '/' + pair + '/*'), 5)

            restored_vp = vp_revelio(vc_wavs)
            cos_distancce = []
            for id in ids:
                vp = vp_dict[id]
                cos_distancce.append(cosine_score(restored_vp, vp))
            cos_rank = pd.Series(cos_distancce).rank(ascending=False)
            atk_index = ids.index(atk_id)
            ranks.append(cos_rank[atk_index])
        except:
            continue

    for top_x in top_xs:
        count = 0
        for _rank in ranks:
            if _rank <= top_x:
                count = count + 1
        topx_rate = count / len(ranks)
        print('Top%d Rate:%.5f'%(top_x, topx_rate))
    return top_xs
    
def osi_dev_clean(vp_dict, testset, ori_path, top_xs, test_path):
    ids = list(vp_dict.keys())
    ranks = []
    for vc_wav in tqdm(testset):
        try:
            clean_id = vc_wav.split('/')[-2]
            clean_wavs = random_select(glob.glob(ori_path + '/' + clean_id + '/*'), 5)
            restored_vp = vp_revelio(clean_wavs)
            cos_distancce = []
            for id in ids:
                vp = vp_dict[id]
                cos_distancce.append(cosine_score(restored_vp, vp))
            cos_rank = pd.Series(cos_distancce).rank(ascending=False)
            clean_index = ids.index(clean_id)
            ranks.append(cos_rank[clean_index])
        except:
            continue

    for top_x in top_xs:
        count = 0
        for _rank in ranks:
            if _rank <= top_x:
                count = count + 1
        topx_rate = count / len(ranks)
        print('Top%d Rate:%.5f'%(top_x, topx_rate))
    return top_xs

def osi_test(test_path, ori_path, top_xs):
    print('#################################')
    print('Starting OSI test')
    vp_dict = osi_vp_cal(test_path, ori_path)
    testset = prase_testfile(test_path)
    if vc_name != 'Clean':
        results = osi_dev(vp_dict, testset, ori_path, top_xs, test_path)
    else:
        results = osi_dev_clean(vp_dict, testset, ori_path, top_xs, test_path)
    return results
    
top_xs = [1,2,3,5,10,20]

threshold = sv_test(test_path, ori_path)
results = osi_test(test_path, ori_path, top_xs)
print(f'{model_name}! {model_src}')
print(f'Test on {vc_name}')
print()
print()
print()
