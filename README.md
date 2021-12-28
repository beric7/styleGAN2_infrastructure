# styleGAN2_infrastructure
### [Training General Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)

## Requirements:
- Windows or Linux (Linux preferred)
- Anaconda3
- Python 3.7
- [Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/)
- Visual Studios Community Edition 2019
- **Make sure to install Microsoft Visual Studios Community Edition before Installing CUDA toolkit**
- Nvidia GPUs with >= 12 GB of memory
- Cuda Toolkit version 11.0, 11.1, or 11.2

## Steps for Installing Requirements and Setting Up the Environment:

**Create the Environment**
```
conda create --name stylegan2-ada-pytorch python=3.7
conda activate stylegan2-ada-pytorch
```

[**Install Visual Studios Community Edition 2019**](https://visualstudio.microsoft.com/vs/)
***Ensure the following are selected when installing Visual Studios***
- Desktop Development with C++
- Universal Windows Platform Development

**Add to PATH** "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

[**CUDA toolkit**](https://developer.nvidia.com/cuda-toolkit-archive) 

***Use Version 11.1/11.2 if using an RTX 3090 series or newer***

**Select the Following on CUDA Installation**
- Windows
- x86_64
- 10
- exe (local)

**Verify CUDA installation by using the following command within anaconda prompt**
```
nvcc --version
```

***Restart Computer after installing CUDA before proceeding***

**Install PyTorch**
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

**Clone the repository and change to the directory**
```
git clone <https_github_link_to_code>
cd <location_of_stylegan2_files>
```

**Install Python Libraries**
```
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
pip install psutil
pip install scipy
```

## Preparing the Data for training

**Gather a training image dataset with one of the following resolutions: 128x128, 256x256, 512x512, 1024x1024**
- Place the images into a directory to be used as the ***source*** flag below
- Create an empty destination directory for the ***dest*** flag which converts the source dataset to be compatible
  with stylegan2-ada-pytorch
```
python dataset_tool.py --source=<source_directory> --dest=<destination_directory>
```

## Training

**Use the following command for training:**
- The ***outdir*** flag should be set to an empty directory. This directory stores checkpoints for the model every 1000 iterations during training.
- The ***data*** flag should be set to the previously created ***dest*** directory for the above data preparation command
- The gpus flag should be set to the number of available gpus for training
- You can clone the generic training folder or our exact training folder we used. Note that you will need to add the DATA and OUTPUT folders for the input data and output model weights respectively. 
```
python train.py --outdir=<out_dir> --data=<destination_directory> --gpus=<desired_number_of_gpus>
```

## Generating with a Pretrained/Trained .pkl Network

Now images can be generated by using the .pkl network with a random noise vector by using ***generate_image_from_noise_z.py OR generate_image_from_noise_w.py***. The z and the w correspond to the different latent spaces which they come from. Typically, we want to utilize the w-space, as it is the less-entangled latent space, meaning that patterns and characteristics within this space are more clearly defined. 

To use these two files properly please note the following section:
```
    network_pkl = "./trained_styleGAN2_models/150k/150k_network-snapshot-025800.pkl"
    
    noise_mode = "random"
    out_dir = "./OUTPUT_DIR/"
    name = 'DIR_NAME'
```
- ***network_pkl*** is the path to the trained model network
- ***noise_mode*** set to random
- ***output_dir*** is the output directory for the images, the csv file of random vectors, and the .npy files.

After running one of these files you will have 10000 randomly generated images from seeds 0-9999. You will have a csv file of 10000 corresponding noise vectors and a npy file of the same 10000 corresponding noise vectors. Depending on which latent space you choose you will have either the z or the w latent space noise vectors. 

