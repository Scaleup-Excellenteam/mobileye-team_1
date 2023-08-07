# This file contains constants.. Mainly strings.
# It's never a good idea to have a string scattered in your code across different files, so just put them here
import os
from pathlib import Path

X = 'x'
Y = 'y'
RED = 'r'
GREEN = 'g'
COLOR = 'color'
SCORE = 'score'
SEQ_IMAG = 'seq_imag'  # Serial number of the image
SEQ_CROP = 'seq_crop'  # Serial number of the attention/crop
I = 'i'
J = 'j'
X0 = 'x0'
Y0 = 'y0'
X1 = 'x1'
Y1 = 'y1'
ZOOM = 'zoom'
PATH = 'path'
FULL_PATH = 'full_path'
CROP_PATH = 'crop_path'
COL = 'col'
LABEL = 'label'
BATCH = 'batch'
IS_TRUE = 'is_true'
IS_IGNORE = 'is_ignore'

TFL_ID = 19  # The pixel value in the labelIds png images


# Data CSV columns:
NAME = 'name'
IMAG_PATH = 'img_path'
C_IMG = 'c_img'
GTIM_PATH = 'gtim_path'
JSON_PATH = 'json_path'
TRAIN_TEST_VAL = 'train_test_val'

# What's in the TRAIN_TEST_VAL column?
TRAIN = 'train'
TEST = 'test'
VALIDATION = 'validation'

# Crop size:
default_crop_w = 32
default_crop_h = 96

base_snc_dir = Path.cwd().parent
DATA_DIR = (base_snc_dir / 'data').as_posix()

attention_results = os.path.join(DATA_DIR, 'attention_results')  # Where we write the results of the attention stage
crops_dir = os.path.join(attention_results, 'crop')  # Where we write the crops
attention_results_h5 = os.path.join(attention_results, 'attention_results_summary.h5')
crop_results_h5 = 'crop_results.h5'
models_dir = os.path.join(DATA_DIR, 'models')  # Where we explicitly copy/save good checkpoints for "release"
logs_dir = os.path.join(models_dir, 'logs')  # Each model will have a folder. TB will show all models

tests_dir = os.path.join(DATA_DIR, 'tests')  # Where each model created a folder with its name, and stores tests data

# File names (directories to be appended automatically)
TFLS_CSV = 'tfls.csv'
ALL_CROPS_CSV = 'all_crops.csv'

