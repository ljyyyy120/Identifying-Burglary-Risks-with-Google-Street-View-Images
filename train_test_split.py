import numpy as np
import os
import shutil
import random

folders = ['01_raw_data/01_manhatten_burglary/01_manhatten_burglary/',
           '01_raw_data/02_brooklyn_burglary/02_brooklyn_burglary/',
           '01_raw_data/03_queens_burglary/03_queens_burglary/',
           '01_raw_data/04_bronx_burglary/04_bronx_burglary/',
           '01_raw_data/05_statenisland_burglary/05_statenisland_burglary/',
           '01_raw_data/06_no_burglary/06_no_burglary/']

train_folder = '01_raw_data/training/'
test_folder = '01_raw_data/testing/'

labels = ['manhattan/', 'brooklyn/', 'queens/', 'bronx/', 'statenisland/', 'no_burglary/']

split = 0.1

os.mkdir(train_folder)
os.mkdir(test_folder)

for l in labels:
    os.mkdir(train_folder + l)
    os.mkdir(test_folder + l)

for fx, f in enumerate(folders):
    data = os.listdir(f)
    n_samples = len(data)
    n_test = int(0.1*n_samples)
    random.shuffle(data)
    test_data = data[:n_test]
    train_data = data[n_test:]
    for t in test_data:
        og_name = f+t
        new_name = test_folder + labels[fx] + t
        shutil.copyfile(og_name, new_name)
    
    for t in train_data:
        og_name = f + t
        new_name = train_folder + labels[fx] + t
        shutil.copyfile(og_name, new_name)

print('fin')