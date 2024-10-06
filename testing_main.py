import os
import math
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import torch.nn as nn
import numpy as np
import argparse
from testing_model import test_model
from model import REDNet20,SRCNN 
from model_utils import str2bool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
  # Create the parser
  parser = argparse.ArgumentParser()

  # Add an argument
  parser.add_argument("--trial_number", type=int, required=True,
                        help="Trial number.")
  parser.add_argument("--model", type=str, required=True,
                        help="The model you want to use.")
  parser.add_argument("--psnr_only", type=str2bool, nargs='?',
                        default=False,
                        help="Compute psnr only, without saving resulting images.")

  # Parse the argument
  args = parser.parse_args()

  # assign the variables according to the argument values
  trialNumber = args.trial_number
  psnr_only = args.psnr_only

  
  if args.model == 'SRCNN':
    model = SRCNN(num_channels=3)
    # model = model_with_upsampling(SRCNN_model)
  elif args.model == 'rednet20':
    model = REDNet20()
  else:
    raise argparse.ArgumentError("Invalid model")

  var_save_dir = './data/variables'
  var_name = 'dataloaders'+'_trial'+str(trialNumber)+'.pkl'
  path = var_save_dir + '/' + var_name
  with open(path, 'rb') as file:
    # Call load method to deserialze
    dataloaders = pickle.load(file)

  model_path_retrieve = "./model/"+"trial"+str(trialNumber)+".pth"

  model.load_state_dict(torch.load(model_path_retrieve))
  model = model.to(device)

  image_saving_dir = './data/data_outs'
  if not os.path.exists(image_saving_dir):
    os.makedirs(image_saving_dir)
  if psnr_only:
    test_model(model,image_saving_dir,dataloaders,psnr_only=True)
  else:
    test_model(model,image_saving_dir,dataloaders)