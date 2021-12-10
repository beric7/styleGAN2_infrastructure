# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:14:12 2021

@author: joshb
"""

import os

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import pandas as pd 



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
    values = []
    for seed_idx, seed in enumerate(seeds):
        
        # This is the randomly generated noise vector
        noise_vector = np.random.RandomState(seed).randn(1, G.z_dim)
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        
        z = torch.from_numpy(noise_vector).to(device)

        
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/seed{seed:04d}.png')
        
        # Dictionary holds info for image names and vector values
        header = f'seed{seed:04d}.png'
        labels.append(header)
        values.append(noise_vector[0])
        dictionary[header] = noise_vector[0];


    df = pd.DataFrame(list(zip(*dictionary.values())), columns = labels)

    df.to_csv('created_images_structural_10000/noise_vectors.csv', index=False, )

import pickle
def generate_image_with_vector(network_pkl,truncation_psi,noise_mode,out_dir, name, optional_vector = None):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device) # type: ignore

    os.makedirs(out_dir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    optional_vector = np.asarray(optional_vector)
    optional_vector = optional_vector.reshape((1,512))
    z = torch.from_numpy(optional_vector).to(device)
    
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/'+name+'result.png')
    
def read_csv_noise_vector(csv_file):
    df = pd.read_csv(csv_file)
    column_names = df.columns
        
    return df, column_names
    
def main():
    network_pkl = "./network-snapshot-013000.pkl"
    #seeds = [33,1000,2000]
    
    npy_file = './interpolations.npy'
    
    noise_mode = "random"
    out_dir = "./interpolations/"

    
    # For generating a single image with manual input vectors (must be a numpy array)
    # vec = np.ones((1,512))
    # generate_image_with_vector(network_pkl, seeds, 0.7, noise_mode, out_dir, vec)
    
    vectors = np.load(npy_file)
    column_name=0
    for vector in vectors:      
        generate_image_with_vector(network_pkl, 0.7, noise_mode, out_dir, str(column_name), vector)
        column_name+=1
    
    
if __name__ == "__main__":
    main()

    