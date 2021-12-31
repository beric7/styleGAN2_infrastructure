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


# Generate multiple images from a randomly generated noise vector
def generate_images(G, device, seeds, truncation_psi, noise_mode, out_dir_z, out_dir_w, out_dir_vec):

    os.makedirs(out_dir_z, exist_ok=True)
    os.makedirs(out_dir_w, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    dictionary_w = {}
    labels_w = []
    dictionary_z = {}
    labels_z = []
    
    # Generate images.
    for seed_idx, seed in tqdm(enumerate(seeds)):
        
        # This is the randomly generated noise vector
        noise_vector_z = np.random.RandomState(seed).randn(1, G.z_dim)
        # z_np = noise_vector_z.cpu().numpy()
        # noise_vector_z = z_np.squeeze(0)[0]
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        
        # save z-space image
        z = torch.from_numpy(noise_vector_z).to(device)
        c = None
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir_z}/z_seed{seed:04d}.png')
        
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        
        # w noise vector
        w_np = w.cpu().numpy()
        noise_vector_w = w_np.squeeze(0)[0]
        
        # save w-space image
        img_w = G.synthesis(w, noise_mode='const', force_fp32=True)
        img_w = (img_w.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img_w[0].cpu().numpy(), 'RGB').save(f'{out_dir_w}/w_seed{seed:04d}.png')
        
        # Dictionary holds info for image names and vector values
        header_w = f'seed_w{seed:04d}.png'
        labels_w.append(header_w)
        dictionary_w[header_w] = noise_vector_w
        
        header_z = f'seed_z{seed:04d}.png'
        noise_vector_z = noise_vector_z.squeeze(0)
        labels_z.append(header_z)
        dictionary_z[header_z] = noise_vector_z

    # W space 
    df_w = pd.DataFrame(list(zip(*dictionary_w.values())), columns = labels_w)
    np_array_w = df_w.to_numpy()
    np.save(out_dir_vec + 'w_space.npy', np_array_w)
    df_w.to_csv(out_dir_vec + 'w_noise_vectors.csv', index=False, )
    
    # Z space 
    df_z = pd.DataFrame(list(zip(*dictionary_z.values())), columns = labels_z)
    np_array_z = df_z.to_numpy()
    np.save(out_dir_vec + 'z_space.npy', np_array_z)
    df_z.to_csv(out_dir_vec + 'z_noise_vectors.csv', index=False, )


def read_csv_noise_vector(csv_file):
    df = pd.read_csv(csv_file)
    column_names = df.columns
        
    return df, column_names
    
def main():
    network_pkl = "./150k_network-snapshot-025800.pkl"
    
    noise_mode = "random"
    out_dir_vec = "./OUTPUT_DIR_VECTORS/"
    out_dir_z = "./OUTPUT_DIR_Z/"
    out_dir_w = "./OUTPUT_DIR_W/"
    
    if not os.path.exists(out_dir_vec): # if it doesn't exist already
        os.makedirs(out_dir_vec)
        
    if not os.path.exists(out_dir_z): # if it doesn't exist already
        os.makedirs(out_dir_z)
        
    if not os.path.exists(out_dir_w): # if it doesn't exist already
        os.makedirs(out_dir_w)
    
    # Generates 1000 random seeds for the dataset
    seeds = []

    for i in range(10):
        num = i
        while num in seeds:
            num = i
        seeds.append(num)
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device)
    
    generate_images(G, device, seeds, 0.7, noise_mode, out_dir_z, out_dir_w, out_dir_vec)

if __name__ == "__main__":
    main()

    
