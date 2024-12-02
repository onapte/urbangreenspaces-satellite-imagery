import os
from pathlib import Path
import torch

TRAIN_DIR = str(Path(os.getcwd()).parents[0])+'/data/original/train'
TEST_DIR = str(Path(os.getcwd()).parents[0])+'/data/original/test'
VAL_DIR = str(Path(os.getcwd()).parents[0])+'/data/original/val'

BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = 0
MODEL_BASE = 'SSD'
IMAGE_SIZE = 300

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
OUT_DIR = str(Path(os.getcwd()).parents[0])+'/geoml_project/output'