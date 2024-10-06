
import os
import math
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model,image_saving_dir,dataloaders,psnr_only=None):
  print("Testing...")
  with torch.no_grad():
    model.eval()
    test_loss = 0.0
    image_id = 0
    for batch,(inputs,targets) in enumerate(tqdm(dataloaders['test'])):
      inputs = inputs.to(device)
      targets = targets.to(device)
      batch_size, Nc, Ny, Nx = inputs.shape

      result = model(inputs)

      # Calculate loss
      criterion = nn.MSELoss()
      loss = criterion(result.float(), targets.float())

      test_loss += loss.item()


      # create the folder for saving output images 
      image_save_folder = image_saving_dir
      if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

      # saves the images and image paths to one same folder if necessary
      if (psnr_only is None) or (psnr_only is False):
        result = Image.fromarray((torch.squeeze(result).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
        targets = Image.fromarray((torch.squeeze(targets).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))

        
        data_str = image_save_folder + '/' + str(image_id) +'_'+'result.TIF'
        label_str = image_save_folder + '/' + str(image_id) +'_'+ 'original.TIF'

        with open(image_saving_dir + '/test_result_paths.txt', 'a+') as txt:
          txt.write(data_str + ' ' + label_str + '\n')

        targets.save(label_str, 'TIFF')
        result.save(data_str, 'TIFF')
      

      image_id = image_id+1

    num_batch = len(dataloaders['test'])
    test_loss /= num_batch
    

    