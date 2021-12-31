# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:14:12 2021

@author: ericb
"""

import os

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import pandas as pd 
from tqdm import tqdm


# this file generates images from a pre-determined noise vector with the option to view in the w or the z space. 
def generate_image_with_vector(G, device, seeds, truncation_psi, noise_mode, out_dir, name, space, optional_vector = None):

    os.makedirs(out_dir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    optional_vector = np.asarray(optional_vector)
    optional_vector = optional_vector.reshape((1,512))
    c = None
    
    if space == 'w':
        optional_vector = np.repeat(optional_vector, repeats=16, axis=0)
        w = torch.from_numpy(optional_vector).to(device).unsqueeze(0)
        img_w = G.synthesis(w, noise_mode='const', force_fp32=True)
        img_w = (img_w.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        name = name.split('.')[0]
        PIL.Image.fromarray(img_w[0].cpu().numpy(), 'RGB').save(f'{out_dir}/'+name+'result.png')
        
    if space == 'z':
        z = torch.from_numpy(optional_vector).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        name = name.split('.')[0]
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/' + name + '_result.png')

def read_csv_noise_vector(csv_file):
    df = pd.read_csv(csv_file)
    column_names = df.columns
        
    return df, column_names
    
def main():
    network_pkl = "./150k_network-snapshot-025800.pkl"
    
    noise_mode = "random"
    out_dir = "./OUTPUT_DIR_REGEN/"
    space = 'w'
    csv_file = './OUTPUT_DIR_VECTORS/w_noise_vectors.csv'
    
    if not os.path.exists(out_dir): # if it doesn't exist already
        os.makedirs(out_dir)
    
    # Generates 1000 random seeds for the dataset
    seeds = []

    for i in range(10000):
        num = i
        while num in seeds:
            num = i
        seeds.append(num)
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device)
    
    df, column_names = read_csv_noise_vector(csv_file) 
    for column_name in tqdm(column_names):
        csv_vector = df[column_name]
        generate_image_with_vector(G, device, seeds, 0.7, noise_mode, out_dir, column_name, space, csv_vector)
    
if __name__ == "__main__":
    main()

    
