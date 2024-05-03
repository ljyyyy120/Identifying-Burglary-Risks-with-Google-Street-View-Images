import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import copy
import os
from random import sample
import random

w = 600
h = 300

folders = ['01_raw_data/01_manhatten_burglary/01_manhatten_burglary/',
           '01_raw_data/02_brooklyn_burglary/02_brooklyn_burglary/',
           '01_raw_data/03_queens_burglary/03_queens_burglary/',
           '01_raw_data/04_bronx_burglary/04_bronx_burglary/',
           '01_raw_data/05_statenisland_burglary/05_statenisland_burglary/',
           '01_raw_data/06_no_burglary/06_no_burglary/']

folders = ['manhattan/', 'brooklyn/', 'queens/', 'bronx/', 'statenisland/', 'no_burglary/']
train_test = ['01_raw_data/training/', '01_raw_data/testing/']

folder_labels = [1, 1, 1, 1, 1, 0]

train_data = [[] for f in folders]
test_data = [[] for f in folders]
for fx, f in enumerate(folders):
    train_data[fx] = os.listdir(train_test[0] + f)
    test_data[fx] = os.listdir(train_test[1]+f)

for ix in range(6):
    for jx in range(len(train_data[ix])):
        train_data[ix][jx] = train_test[0] + folders[ix] + train_data[ix][jx]
    for jx in range(len(test_data[ix])):
        test_data[ix][jx] = train_test[1] + folders[ix] + test_data[ix][jx]

def calc_conv_output_shape(n1, n2, k1, k2=None):
    if k2 is None:
        k2 = k1
    
    d1 = n1-k1+1
    d2 = n2 - k2 + 1
    return (d1, d2)

def calc_pooling_output_shape(n1, n2, k1, k2=None):
    if k2 is None:
        k2 = k1
    d1 = (n1-k1-1-1)/k1 +1
    d2 = (n2-k2-1-1)/k2 +1
    return (int(np.ceil(d1)), int(np.ceil(d2)))


class CNN_Logistic_Regressor(nn.Module):
    def __init__(self, n1 = 300, n2= 600, ch = 3, ksize=[21,11,5], out_channels=[6,12,18], pooling_kernel=[5, 3, 3], fc_out=[120, 60, 10]):
        super().__init__()
        self.conv = []
        self.fc = []
        self.conv.append(nn.Conv2d(in_channels=3, out_channels=out_channels[0], kernel_size=ksize[0]))
        self.ksize = ksize
        self.out_channels = out_channels
        self.pooling_kernels = pooling_kernel
        self.fc_out = fc_out
        dims = (n1, n2)
        dims = calc_conv_output_shape(*dims, ksize[0])
        dims = calc_pooling_output_shape(*dims, pooling_kernel[0])
        for ix in range(1,len(ksize)):
            self.conv.append(nn.Conv2d(in_channels=out_channels[ix-1], out_channels=out_channels[ix], kernel_size=ksize[ix]))
            dims = calc_conv_output_shape(*dims, ksize[ix])
            dims = calc_pooling_output_shape(*dims, pooling_kernel[ix])
        
        self.fc_input = dims[0]*dims[1]*out_channels[-1]
        print(self.fc_input)
        self.fc.append(nn.Linear(in_features=self.fc_input, out_features=fc_out[0]))
        for ix in range(1,len(fc_out)):
            self.fc.append(nn.Linear(in_features=fc_out[ix-1],out_features=fc_out[ix]))
        self.final_fc = nn.Linear(in_features=fc_out[-1],out_features=1)
        self.out = nn.Sigmoid()
    
    def forward(self,t):
        x = copy.copy(t)
        for ix in range(len(self.conv)):
            x = self.conv[ix](x)
            x = F.relu(x)
            x = F.max_pool2d(x,kernel_size=self.pooling_kernels[ix])
        x = x.reshape(-1,self.fc_input)
        for ix in range(len(self.fc)):
            x = self.fc[ix](x)
            x = F.relu(x)
        x = self.final_fc(x)
        x = self.out(x)
        return x

class Data_Loader():
    def __init__(self, batch_size, borough=None, only_burglary=False, training=True):
        self.batch_size = batch_size
        self.borough=borough
        self.only_burglary=only_burglary
        if training is True:
            self.data=train_data
        else:
            self.data = test_data
        for ix in range(6):
            random.shuffle(self.data[ix])
        n_per = np.zeros(6,dtype=int)
        if only_burglary is True:
            if only_burglary is True:
                if borough is None:
                    n_per[:-1] = batch_size//5
                    bx = 0
                    for ix in range(batch_size % 5):
                        n_per[bx] += 1
                        bx += 1
                else:
                    n_per[borough] = batch_size
        elif borough is not None:
            n_per[borough] = batch_size//2 + batch_size%2
            n_per[-1] = batch_size//2
        else:
            n_per[-1] = batch_size//2
            n_burg = batch_size - batch_size//2
            n_per[:-1] = n_burg//5
            bx = 0
            for ix in range(n_burg%5):
                n_per[bx] += 1
                bx +=1
        self.n_per = n_per
        self.indices = [0 for i in range(6)]
        self.max_index = [len(i) for i in self.data]
        
        self.eval_finished = False
        self.eval_started = False
    
    def reset(self):
        self.eval_finished = False
        self.eval_started = False
        self.indicies = [0 for i in range(6)]
    
    def get_sample(self):
        samples = []
        labels = []
        for ix in range(6):
            n = self.n_per[ix]
            samples.extend(sample(self.data[ix],n))
            for jx in range(n):
                labels.append(folder_labels[ix])

        zipped = list(zip(samples,labels))
        random.shuffle(zipped)
        sample_sh, labels_sh = zip(*zipped)
        sample_sh = list(sample_sh)
        labels_sh = list(labels_sh)
        labels_sh_tensor = torch.FloatTensor(labels_sh)
        img = cv2.imread(sample_sh[0])
        img_tens = transforms.functional.to_tensor(img).unsqueeze(0)
        for i in range(1,len(sample_sh)):
            img = cv2.imread(sample_sh[i])
            temp = transforms.functional.to_tensor(img)
            img_tens = torch.concat((img_tens,temp.unsqueeze(0)))
        return img_tens, labels_sh_tensor
    
    def get_sample_specific_label_and_borough(self, borough):
        samples = []
        labels = []
        samples.extend(sample(self.data[borough],self.batch_size))
        for ix in range(self.batch_size):
            labels.append(folder_labels[borough])
        
        img = cv2.imread(samples[0])
        img_tens = transforms.functional.to_tensor(img).unsqueeze(0)
        for i in range(1,len(samples)):
            img = cv2.imread(samples[i])
            temp = transforms.functional.to_tensor(img)
            img_tens = torch.concat((img_tens,temp.unsqueeze(0)))
        label_tens = torch.FloatTensor(labels)
        return img_tens, label_tens
    
    def get_samples_for_eval(self, borough):
        if self.eval_started == False:
            self.indices[borough] = 0
            self.eval_started = True
        if self.indices[borough]+self.batch_size > self.max_index[borough]:
            ix = self.max_index[borough]
            self.eval_finished = True
            self.eval_started = False
        else:
            ix = self.indices[borough] + self.batch_size
        samples = self.data[borough][self.indices[borough]:ix]

        n_labels = ix - self.indices[borough]
        self.indices[borough] += self.batch_size
        img = cv2.imread(samples[0])
        img_tens = transforms.functional.to_tensor(img).unsqueeze(0)
        for i in range(1,len(samples)):
            img = cv2.imread(samples[i])
            temp = transforms.functional.to_tensor(img)
            img_tens = torch.concat((img_tens,temp.unsqueeze(0)))
        if borough != 5:
            labels = torch.ones(n_labels)
        else:
            labels = torch.zeros(n_labels)
        return img_tens, labels, self.eval_finished
    
def calc_num_right(labels, preds):
    return np.sum(labels.numpy()==torch.round(preds.detach()).flatten().numpy())

def evaluate_model(model, borough, training=False):
    dL = Data_Loader(32,borough=borough, training=training)
    n_right_b = 0
    n_tot_b = 0
    n_right_nb = 0
    n_tot_nb = 0
    finished = 0

    while finished == 0:
        imgs, labels, finished = dL.get_samples_for_eval(borough)
        pred = model(imgs)
        n_tot_b += len(labels)
        n_right_b += calc_num_right(labels, pred)
    
    finished = 0
    dL.reset()
    while finished == 0:
        imgs, labels, finished = dL.get_samples_for_eval(5)
        pred = model(imgs)
        n_tot_nb += len(labels)
        n_right_nb += calc_num_right(labels, pred)
    
    acc_b = n_right_b/n_tot_b
    acc_nb = n_right_nb/n_tot_nb
    acc_tot = (acc_b + acc_nb)/2
    return acc_tot, acc_b, acc_nb, n_tot_b, n_tot_nb

class Model_Trainer():
    def __init__(self, borough, epochs=20, steps_per_epoch=500, lr = 0.001, conv_ksize=[3,3,3,3], conv_out_ch=[16,32,64,128], pool_kernel=[3,3,3,3], fc_out=[256], batch_size=32):
        self.borough = borough
        self.epochs = epochs
        self.steps = steps_per_epoch
        self.batch_size=32
        self.data_loader = Data_Loader(batch_size=batch_size, borough=borough)
        self.cnn = CNN_Logistic_Regressor(ksize=conv_ksize, out_channels=conv_out_ch, pooling_kernel=pool_kernel, fc_out=fc_out)
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), 0.001)
        self.losses = []
        self.min_loss = np.inf
        self.val_acc = 0.0
    
    def train(self):
        for epoch in tqdm(range(self.epochs), 'Epoch No: '):
            for step in tqdm(range(self.steps), 'Step No: '):
                if step == self.steps-1:
                    acc_tot, _, _, _, _ = evaluate_model(self.cnn, self.borough)
                    if acc_tot > self.val_acc:
                        torch.save(self.cnn, 'optimal_model_validated_borugh_{}.pth'.format(self.borough))
                        print('\nmaximum test accuracy: ', acc_tot)
                        self.val_acc = acc_tot
                imgs, labels = self.data_loader.get_sample()
                self.optimizer.zero_grad()
                preds = self.cnn(imgs)
                loss = self.loss_function(preds.flatten(), labels)
                if loss < self.min_loss:
                    if epoch < 7:
                        torch.save(self.cnn, 'optimal_model_train_borough_{}.pth'.format(self.borough))
                        self.min_loss = loss
                        print('\nminimum loss: ', self.min_loss)
                    else:
                        acc_tot, _, _, _, _ = evaluate_model(self.cnn, self.borough)
                        if acc_tot > self.val_acc:
                            torch.save(self.cnn, 'optimal_model_validated_borugh_{}.pth'.format(self.borough))
                            print('\nmaximum test accuracy: ', acc_tot)
                            self.val_acc = acc_tot
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.detach())
        
        torch.save(self.cnn, 'Last_step_model_borough{}.pth'.format(self.borough))

    def load_model(self, file_name):
        self.cnn = torch.load(file_name)
        self.cnn.eval()

    def eval_OS(self):
        acc_tot, acc_b, acc_nb, n_tot_b, n_tot_nb = evaluate_model(self.cnn, self.borough)
        return acc_tot, acc_b, acc_nb, n_tot_b, n_tot_nb

    def eval_IS(self):
        acc_tot, acc_b, acc_nb, n_tot_b, n_tot_nb = evaluate_model(self.cnn, self.borough, training=True)
        return acc_tot, acc_b, acc_nb, n_tot_b, n_tot_nb


