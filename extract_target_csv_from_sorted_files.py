# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:13:26 2021

@author: Admin
"""

import pandas as pd
from build_image_file_list import build_image_file_list_sorted

target_file_dir = './random_noise_images_1000/corroded_100/'

df = pd.read_csv('./random_noise_images_1000/noise_vectors_1000.csv')

imageFilePaths, image_names = build_image_file_list_sorted(target_file_dir)

dict_target = {}
for name in image_names:
    temp_list = df[name].tolist()
    dict_target[name] = (temp_list)

target_df = pd.DataFrame(dict_target)

target_df.to_csv('./random_noise_images_1000/corrosion_vectors_1000.csv')



