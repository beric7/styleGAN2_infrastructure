# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:14:12 2021

@author: joshb and ericb
"""

import os

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import pandas as pd 
from tqdm import tqdm


# Generate multiple images from a randomly generated noise vector
def generate_images(network_pkl, seeds,truncation_psi,noise_mode,out_dir):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    '''
    with dnnlib.util.open_url(network_pkl) as f:
    '''
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device)

    os.makedirs(out_dir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    dictionary = {}
    labels = []
    for seed_idx, seed in tqdm(enumerate(seeds)):
        
        # This is the randomly generated noise vector
        noise_vector = np.random.RandomState(seed).randn(1, G.z_dim)
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        
        z = torch.from_numpy(noise_vector).to(device)
        z_np = z.cpu().numpy()
        c = None
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        w_np = w.cpu().numpy()
        noise_vector_w = w_np.squeeze(0)[0]
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        img_w = G.synthesis(w, noise_mode='const', force_fp32=True)
        img_w = (img_w.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img_w[0].cpu().numpy(), 'RGB').save(f'{out_dir}/w_seed{seed:04d}.png')
        
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/seed{seed:04d}.png')
        
        # Dictionary holds info for image names and vector values
        header = f'seed{seed:04d}.png'
        labels.append(header)
        dictionary[header] = noise_vector;
    
    
    
    df = pd.DataFrame(list(zip(*dictionary.values())), columns = labels)

    df.to_csv('created_images_structural_585/noise_vectors.csv', index=False, )

def generate_noise_vectors(network_pkl, seeds,truncation_psi,noise_mode,out_dir, name):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    '''
    with dnnlib.util.open_url(network_pkl) as f:
    '''
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device)

    os.makedirs(out_dir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    dictionary = {}
    labels = []
    for seed_idx, seed in tqdm(enumerate(seeds)):
        
        # This is the randomly generated noise vector
        noise_vector = np.random.RandomState(seed).randn(1, G.z_dim)
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        
        z = torch.from_numpy(noise_vector).to(device)
        c = None
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        w_np = w.cpu().numpy()
        noise_vector_w = w_np.squeeze(0)[0]

        # Dictionary holds info for image names and vector values
        header = f'seed{seed:04d}.png'
        labels.append(header)
        dictionary[header] = noise_vector_w
    
    
    df = pd.DataFrame(list(zip(*dictionary.values())), columns = labels)
    np_array = df.to_numpy()
    
    np_array = np_array

    np.save(out_dir + name + '.npy', np_array)

    df.to_csv(out_dir + 'noise_vectors.csv', index=False, )
    
def read_csv_noise_vector(csv_file):
    df = pd.read_csv(csv_file)
    column_names = df.columns
        
    return df, column_names
    
def main():
    network_pkl = "./trained_styleGAN2_models/culverts/network-snapshot-013000.pkl"
    #seeds = [33,1000,2000]
    # pickle = legacy.load_network_pkl(network_pkl)
    csv_file = './585_noise_vector_w.csv'
    
    noise_mode = "random"
    out_dir = "./585_noise_vector_w/"
    name = 'corrosion_w'
    
    if not os.path.exists(out_dir): # if it doesn't exist already
        os.makedirs(out_dir)
    
    # Generates 1000 random seeds for the dataset
    seeds = []

    for i in range(10000):
        randomNum = i
        while randomNum in seeds:
            randomNum = i
        seeds.append(randomNum)
    
    seeds = [585]
    # For generating multiple images from randomly generated vectors
    generate_noise_vectors(network_pkl, seeds,0.7,noise_mode,out_dir, name)
    
if __name__ == "__main__":
    main()

    
