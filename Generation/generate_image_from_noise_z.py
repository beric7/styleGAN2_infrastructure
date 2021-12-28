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
        z_np = z.cpu().numpy()
        c = None
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        w_np = w.cpu().numpy()
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
        values.append(noise_vector[0])
        dictionary[header] = noise_vector[0];


    df = pd.DataFrame(list(zip(*dictionary.values())), columns = labels)

    df.to_csv(out_dir + '/noise_vectors.csv', index=False, )

import pickle
def generate_image_with_vector(network_pkl, seeds,truncation_psi,noise_mode,out_dir, name, optional_vector = None):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    object_ = pd.read_pickle(network_pkl)
    G = object_['G_ema'].to(device) # type: ignore

    os.makedirs(out_dir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    optional_vector = np.asarray(optional_vector).T
    optional_vector = optional_vector.reshape((1,512))
    z = torch.from_numpy(optional_vector).to(device)
    c = None
    w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
    img_w = G.synthesis(w, noise_mode='const', force_fp32=True)
    img_w = (img_w.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img_w[0].cpu().numpy(), 'RGB').save(f'{out_dir}/'+name+'result.png')
    
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/'+name+'result.png')
    PIL.Image.fromarray(img_w[0].cpu().numpy(), 'RGB').save(f'{out_dir}/'+name+'result.png')
    
def read_csv_noise_vector(csv_file):
    df = pd.read_csv(csv_file)
    column_names = df.columns
        
    return df, column_names
    
def main():
    network_pkl = "./network-snapshot-013000.pkl"
    #seeds = [33,1000,2000]
    # pickle = legacy.load_network_pkl(network_pkl)
    csv_file = './created_images_structural_test.csv'
    
    noise_mode = "random"
    out_dir = "./created_images_structural_z/"
    
    # Generates 1000 random seeds for the dataset
    seeds = []
    '''
    for i in range(10000):
        randomNum = np.random.randint(99999)
        while randomNum in seeds:
            randomNum = np.random.randint(99999)
        seeds.append(randomNum)
    '''
    for i in range(10000):
        randomNum = i

        seeds.append(randomNum)

    
    # For generating multiple images from randomly generated vectors
    generate_images(network_pkl, seeds,0.7,noise_mode,out_dir)
    
    # For generating a single image with manual input vectors (must be a numpy array)
    # vec = np.ones((1,512))
    # generate_image_with_vector(network_pkl, seeds, 0.7, noise_mode, out_dir, vec)
    
    df, column_names = read_csv_noise_vector(out_dir + '/noise_vectors.csv') 
    for column_name in column_names:
        csv_vector = df[column_name]
        generate_image_with_vector(network_pkl, seeds, 0.7, noise_mode, out_dir, column_name, csv_vector)
    
    
if __name__ == "__main__":
    main()

    
