# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:12:10 2021

@author: ericb
"""
from skimage import metrics as metric
from skimage import measure as measure
import cv2
from PIL import Image
import os

test_dir = './candidate_real_embedded/150k_plus/image/'
true_dir = './candidate_real_embedded/150k_plus/image/'

PeakSig = 0
StrucSim = 0
count = 0

for image in os.listdir(true_dir):
    
    true = true_dir + image
    true = cv2.imread(true)
    
    test = test_dir + image
    test = cv2.imread(test)
    
    psnr = metric.peak_signal_noise_ratio(true, test)
    ssim = metric.structural_similarity(true, test, multichannel=True)
    
    PeakSig = PeakSig + psnr
    StrucSim = StrucSim + ssim
    count = count + 1
    
PeakSig = PeakSig / count

StrucSim = StrucSim / count

print('peak signal ratio {}', PeakSig)
print('Structural Similiarity {}', StrucSim)
