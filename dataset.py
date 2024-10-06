#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as data
from torchvision.transforms import ToPILImage 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from tqdm import tqdm
import time

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import pdb
#%%

BLOCK_SIZE = 64  
IMAGE_SIZE = 512,512

class ToBayer(object):
    def remosaic(self,img):
        Nc, Ny, Nx = img.shape
        R = np.zeros([2*Ny, 2*Nx])
        G = np.zeros([2*Ny, 2*Nx])
        B = np.zeros([2*Ny, 2*Nx])
        # R_mask = -1*np.ones([2*Ny, 2*Nx])
        for i in range(1,Ny):
            for j in range(1,Nx):
                R[2*i-1,2*j-1] = img[0,i,j]
                # R_mask[2*i-1,2*j-1] = 0
                B[2*i,2*j] = img[2,i,j]
                G[2*i,2*j-1] = img[1,i,j]
                G[2*i-1,2*j] = img[1,i,j]



        return np.array([R,G,B])

    def __call__(self, pic):
        
        return self.remosaic(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# taken from https://github.com/GitHberChen/Deep-Residual-Network-for-JointDemosaicing-and-Super-Resolution/blob/a224f0ea673d70c26ee17aec9f27e1a7c31cbe8e/img_to_imgblk.py
def To_Bayer(img):
    w, h = img.size
    img = img.resize((int(w), int(h)), resample=Image.Resampling.LANCZOS)
    w, h = img.size
    # r,g,b=img.split()
    data = np.array(img)
    """
    R G R G
    G B G B
    R G R G
    G B G B
    """
    bayer_mono = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            if (0 == r % 2):
                if (1 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 1] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 0]
            else:
                if (0 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 0] = 0
                    data[r, c, 1] = 0

                    bayer_mono[r, c] = data[r, c, 2]

    # Three channel Bayer image
    bayer = Image.fromarray(data)
    # bayer.show()

    return bayer


# resample the original RGB image into three subsamples
class Resampling(object):
    # def __init__(self,img):
    #   self.img = img

    def resample(self,img,downsize=False):
        Nc, Ny, Nx = img.shape
        R = np.zeros([Ny, Nx])
        G = np.zeros([Ny, Nx])
        B = np.zeros([Ny, Nx])

        # green channel
        for i in range(0,Ny,1): #row
          for j in range(0,Nx,2): #column
            if (i%2)==0: # even rows
              if (j+1)<=Nx-1:
                G[i,j+1] = img[1,i,j+1]
            elif (i%2)==1: # odd rows
              G[i,j] = img[1,i,j]

        # red channel and blue channel
        for i in range(0,Ny,2):
          for j in range(0,Nx,2):
            B[i,j] = img[2,i,j] # blue channel
            if (i+1)<=Ny-1 and (j+1)<=Nx-1:
              R[i+1,j+1] = img[0,i+1,j+1] # red channel
        
        # self.resampled = np.array([R,G,B])
        # print("RGB resampled: ", np.array([R,G,B]))
        return torch.from_numpy(np.array([R,G,B]))

    def __call__(self, img):
        
        return self.resample(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# resample the original RGB image into a Bayer image
class Resampling_Bayer(object):
    # def __init__(self,img):
    #   self.img = img

    def resample_bayer(self,img):
        Nc, Ny, Nx = img.shape
        bayer = np.zeros([Ny, Nx])

        # green channel
        for i in range(0,Ny,1): #row
          for j in range(0,Nx,2): #column
            if (i%2)==0: # even rows
              if (i+1)<=Ny-1 and (j+1)<=Nx-1:
                bayer[i,j+1] = img[1,i,j+1] # prevents breaking if image size does not allow another green pixel
            elif (i%2)==1: # odd rows
              bayer[i,j] = img[1,i,j]

        # red channel and blue channel
        for i in range(0,Ny,2):
          for j in range(0,Nx,2):
            bayer[i,j] = img[2,i,j] # blue channel
            if (i+1)<=Ny-1 and (j+1)<=Nx-1: # prevents breaking if image size does not allow another red pixel
              bayer[i+1,j+1] = img[0,i+1,j+1] # red channel
        
        return torch.from_numpy(bayer)

    def __call__(self, img):
        
        return self.resample_bayer(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

#%%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        Resampling()
        # ToBayer()

     ])

transform2 = transforms.Compose(
    [
        transforms.ToTensor()

     ])

transform3 = transforms.Compose(
    [
        transforms.ToTensor(),
        Resampling_Bayer()

     ])

def preprocess_Dataset(save_path, txt_file_path,img_dir):
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  image_info = pd.read_csv(txt_file_path, delim_whitespace=True,header=None,names=['id','path'])
  time_start = time.perf_counter()

  for i in tqdm(range(image_info.shape[0])):
    image_path = image_info.iloc[i,1]
    image_id = image_info.iloc[i,0]
    image_full_path = img_dir + '/' + image_path
    img = Image.open(image_full_path)
    # only proceed if image has three channels
    if np.array(img).ndim ==3:
      img = img.resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS) # resize image to 512x512
      w, h = img.size
      row = h // BLOCK_SIZE
      col = w // BLOCK_SIZE
      print("Image_id:", image_id, "WxH=", w, 'x', h)
      patch_id = 0
      for r in range(row):
        for c in range(col):
          left = c * BLOCK_SIZE
          upper = r * BLOCK_SIZE
          right = left + BLOCK_SIZE
          lower = upper + BLOCK_SIZE
          temp = img.crop((left, upper, right, lower)).convert('RGB')

          temp_bayer = To_Bayer(temp)
          image_save_folder = save_path + '/' + str(image_id)
          if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)

          data_str = save_path + '/' + str(image_id) + '/' + str(image_id) +'_'+ str(patch_id)+'_'+'data.TIF'
          label_str = save_path + '/' + str(image_id) + '/' + str(image_id) +'_'+ str(patch_id)+'_'+ 'label.TIF'
          with open('./data/paths.txt', 'a+') as txt: # save paths.txt in data folder instead
          # with open(save_path + '/paths.txt', 'a+') as txt:
            txt.write(data_str + ' ' + label_str + '\n')

          temp.save(label_str, 'TIFF')
          temp_bayer.save(data_str, 'TIFF')
          patch_id = patch_id+1
  time_used = time.perf_counter() - time_start
  print('Time used:', time_used)

def preprocess_Dataset_no_crop(save_path, txt_file_path,img_dir):
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  image_info = pd.read_csv(txt_file_path, delim_whitespace=True,header=None,names=['id','path'])
  time_start = time.perf_counter()

  for i in tqdm(range(image_info.shape[0])):
    image_path = image_info.iloc[i,1]
    image_id = image_info.iloc[i,0]
    image_full_path = img_dir + '/' + image_path
    img = Image.open(image_full_path)
    
    # only proceed if image has three channels
    if np.array(img).ndim ==3:
      img = img.resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS) # resize image to 512x512
      
      temp_bayer = To_Bayer(img)
      # pdb.set_trace()
      data_str = save_path + '/' + str(image_id) +'_'+'data.TIF'
      label_str = save_path + '/' + str(image_id) +'_'+ 'label.TIF'
      with open('./data/paths_no_crop.txt', 'a+') as txt: # save paths_no_crop.txt in data folder instead
      # with open(save_path + '/paths.txt', 'a+') as txt:
        txt.write(data_str + ' ' + label_str + '\n')

      img.save(label_str, 'TIFF')
      temp_bayer.save(data_str, 'TIFF')

  time_used = time.perf_counter() - time_start
  print('Time used:', time_used)

# code taken from https://github.com/GitHberChen/Deep-Residual-Network-for-JointDemosaicing-and-Super-Resolution/blob/a224f0ea673d70c26ee17aec9f27e1a7c31cbe8e/DataSet.py#L15
def bayer2mono(bayer_img):
    bayer_img = np.array(bayer_img)
    bayer_mono = np.max(bayer_img, 2)
    return bayer_mono[:, :, np.newaxis]

class Img_Dataset(Dataset):
    def __init__(self, img_paths,transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_path: path to a text file containing paths to each image and bayer 
        :param three_channel_bayer: True or False

        """
        self.img_paths = pd.read_csv(img_paths, delim_whitespace=True,header=None)
        # self.transform = transform

    def get_image_from_folder(self, path):
        """
        gets a image by a name gathered from file list text file

        :param path: path of targeted image
        :return: a PIL image
        """

        image = Image.open(path)

        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return self.img_paths.shape[0]

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        img_original = self.get_image_from_folder(self.img_paths.iloc[index,1]).convert('RGB')

        img_bayer = bayer2mono(self.get_image_from_folder(self.img_paths.iloc[index,0]).convert('RGB'))

        # convert to tensor in range 0-1
        transform1 = transforms.Compose([transforms.ToTensor()])
       

        img_original_1 = transform1(img_original)
        img_bayer_1 = transform1(img_bayer)  

        return img_bayer_1.float(),img_original_1.float()

class Dataset_threeBayerChannel(Dataset):
    def __init__(self, img_paths, three_channel_bayer=False, transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_path: path to a text file containing paths to each image and bayer 
        :param three_channel_bayer: True or False

        """
        self.img_paths = pd.read_csv(img_paths, delim_whitespace=True,header=None)

        self.three_channel_bayer = three_channel_bayer

    def get_image_from_folder(self, path):
        """
        gets a image by a name gathered from file list text file

        :param path: path of targeted image
        :return: a PIL image
        """

        image = Image.open(path)

        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return self.img_paths.shape[0]

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        img_original = self.get_image_from_folder(self.img_paths.iloc[index,1]).convert('RGB')
 
        if not self.three_channel_bayer:
          # bayer input to model will be of 1 channel only
          img_bayer = bayer2mono(self.get_image_from_folder(self.img_paths.iloc[index,0]).convert('RGB'))
        else:
          # bayer input to model will be of 3 channels
          img_bayer = self.get_image_from_folder(self.img_paths.iloc[index,0]).convert('RGB')

        # convert to tensor in range 0-1
        transform1 = transforms.Compose([transforms.ToTensor()])
        # transform2 = transforms.Compose([transforms.ToTensor(),AddGaussianNoise(0., 1.)])

        img_original_1 = transform1(img_original)
        img_bayer_1 = transform1(img_bayer)
        return img_bayer_1.float(),img_original_1.float()

    
def get_data_loader(three_channel_bayer=None,no_crop=None,batch_size=None,transform=None):
    if no_crop is None:
      img_paths = "./data/paths.txt"
    elif no_crop is True:
      img_paths = "./data/paths_no_crop.txt"

    if batch_size is None:
      batch_size = 16
    else:
      batch_size = batch_size
    validation_split = 0.2
    test_split = 0.1
    shuffle_dataset = True
    random_seed= 42
    
    # eliminating randomness
    worker_init_fn=np.random.seed(random_seed)
    # numpy vars
    np.random.seed(random_seed)
    # Pytorch vars
    torch.manual_seed(random_seed)
    # cuda
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    

    # create dataset
    if three_channel_bayer is not None:
      if transform is not None:
        Bayer_dataset = Dataset_threeBayerChannel(img_paths=img_paths,three_channel_bayer=three_channel_bayer,transform=transform)
      else:
        Bayer_dataset = Dataset_threeBayerChannel(img_paths=img_paths,three_channel_bayer=three_channel_bayer)
    else:
      if transform is not None:
        Bayer_dataset = Img_Dataset(img_paths=img_paths,transform=transform)
      else:
        Bayer_dataset = Img_Dataset(img_paths=img_paths)
    # Creating data indices for training and validation splits:
    dataset_size = len(Bayer_dataset)
    indices = list(range(dataset_size))
    val_split_end = int(np.floor(validation_split * dataset_size))
    test_split_end = val_split_end+int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    val_indices, test_indices, train_indices = indices[:val_split_end], indices[val_split_end:test_split_end], indices[test_split_end:]

    # Creating data samplers and loaders:

    # create the test subset explicitly because it needs to be sampled in a particular order
    dataset_test = torch.utils.data.Subset(Bayer_dataset, test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SequentialSampler(dataset_test)
    # print("train_indices: ",train_indices)
    # print("val_indices: ",val_indices)
    # print("test_indices: ",test_indices)

    train_dataloader = torch.utils.data.DataLoader(Bayer_dataset, batch_size=batch_size, 
                                                  sampler=train_sampler, num_workers=0,worker_init_fn=worker_init_fn)
    valid_dataloader = torch.utils.data.DataLoader(Bayer_dataset, batch_size=batch_size, 
                                                  sampler=valid_sampler, num_workers=0,worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(Bayer_dataset, batch_size=1, 
                                                  sampler=test_sampler, num_workers=0,worker_init_fn=worker_init_fn)
    
    return {'train':train_dataloader, 'val':valid_dataloader, 'test':test_dataloader}



