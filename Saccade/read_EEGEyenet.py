import os
import sys
import glob
import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
from read_data import * 
from tqdm import tqdm
import ipdb
import math


def resample_freq(eeg_data, old_freq, new_freq):
    t_total = eeg_data.shape[1] * 1./old_freq
    
    xnew = np.arange(0, t_total, 1./new_freq)
    xold = np.arange(0, t_total, 1./old_freq)
    
    eeg_data_new = np.zeros((eeg_data.shape[0], xnew.shape[0]))
    for i in range(eeg_data.shape[0]):
        eeg_data_new[i, :] = np.interp(xnew, xold, eeg_data[i, :])
    return eeg_data_new
            

def split_stratified(block_id, kfold):
    np.random.seed(24)
    
    train_list, test_list = [], []
    
    
    for (srlno, bid) in enumerate(np.unique(block_id)):
        curr_trails = np.where(block_id == bid)[0]
        np.random.shuffle(curr_trails)
        
        n_trials_per_fold = int(np.ceil(curr_trails.shape[0]/kfold))
        
        #j gives the fold
        for j in range(kfold):
            
            if srlno == 0:
                train_list.append([])
                test_list.append([])
            
            temp_test_fold = []
            temp_train_fold = []   
            #k is used to split the curr_trials accordingly
            for k in range(kfold):
                if k==kfold - 1:
                    ed = curr_trails.shape[0]
                else:
                    ed = (k+1)*n_trials_per_fold
                #test case
                if k == j:
                    temp_test_fold = curr_trails[k*n_trials_per_fold:ed]
                #train case
                else:
                    temp_train_fold.append(curr_trails[k*n_trials_per_fold:ed])
            
            
            train_list[j] += list(np.concatenate(temp_train_fold))
            test_list[j] += list(temp_test_fold)
    
    for i in range(kfold):
        np.random.shuffle(train_list[i])
        np.random.shuffle(test_list[i])
    #print(np.sum(block_id == bid))
    #print(all_trials.shape)
    #all_trials= convert_for_seqNets(all_trials)
    #print(all_trials.shape)
    return train_list, test_list, test_list

def split_benchmark(ids, train, val, test, fid='DDir', perc_data=1.0):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    if fid == 'DDir':
        test_split = 27
        val_split = 28    
        train_split = int(120*perc_data)
        train_split_full = 120
    else:
       test_split = math.ceil(test * num_ids)
       val_split = math.ceil(val * num_ids)
       train_split = int((num_ids - val_split - test_split)*perc_data)
       train_split_full = num_ids - val_split - test_split
    
    # print(val_split, test_split, train_split)
    #sys.exit()
    train_full = np.isin(ids, IDs[:train_split_full])

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split_full+val_split:])
    print(np.sum(train), np.sum(val), np.sum(test))
    return train, val, test, train_full
    
def read_data_EyeNet(dat_id):
    
    #all files have 'EEG' and 'labels' as the files in the npz
    #EEG -> of the shape (N, 500, 129) ---> 1 sec window.
    #labels[:, 0]: subject id
    if dat_id == 'LR':
        #left-right antisaccade data
        #labels -> (N, 2), 2nd column left (0)/right (1); N = 30842 
        data_file = np.load("./data_EEGEyeNet/LR_task_with_antisaccade_synchronised_min.npz")
        
    elif dat_id == 'PS':
        #processing speed; visual search
        #smallest theta
        #labels -> (N, 3), 2, 3rd column is del_r (in mm), del_theta (relative angle) ; N = 31563
        data_file = np.load("./data_EEGEyeNet/Direction_task_with_processing_speed_synchronised_min.npz")
        
    elif dat_id == 'DPos':
        #dots: position
        #labels -> (N, 3), 2, 3rd column is del_x, del_y (relative position) during saccade; N = 21464 
        data_file = np.load("./data_EEGEyeNet/Position_task_with_dots_synchronised_min.npz")
        
    elif dat_id == 'DDir':
        #dots: direction
        #labels -> (N, 3), 2, 3rd column is del_r (in mm), del_theta (relative angle); N = 17830
        data_file = np.load("./data_EEGEyeNet/Direction_task_with_dots_synchronised_min.npz")
            
    else:
        sys.exit()
    
    return np.expand_dims(np.transpose(data_file['EEG'][:, :, 0:128], axes=(0,2,1)), axis=-1).astype('float32'), data_file['labels']
    
def data_reader_wrapper(fid, perc_data=1.0):
    eeg_data, eye_data = read_data_EyeNet(fid)
    # print(eeg_data.shape, eye_data.shape)
    ids = eye_data[:, 0].astype('int')
    #print(np.unique(ids))
    #train_split, val_split, test_split = split_stratified(ids, 5) 
    train_split, val_split, test_split, train_split_full = split_benchmark(ids, 0.7, 0.15, 0.15, fid, perc_data)
    # print(np.min(eye_data[:, 1]), np.max(eye_data[:, 1]), np.min(eye_data[:, 2]), np.max(eye_data[:, 2]))
    # print(np.sum(train_split), np.sum(test_split))
    #print(eye_data)
    #return eeg_data, eye_data[:, 1:], train_split[0], val_split[0], test_split[0]
    # from deepak_eye_dat_prep import prepEyeTracking
    # eye_proc_obj = prepEyeTracking()
    # eye_proc_obj.plot_velocity_hist(eye_data[:, 1], False, 100, "r (in pix)", "Density", "del r", "./Results/EEGNet_LSTM_WM/stats/del_r_eyenet.png")
    # eye_proc_obj.plot_velocity_hist(eye_data[:, 2], False, 100, "theta (in radians)", "Density", "del theta", "./Results/EEGNet_LSTM_WM/stats/del_theta_eyenet.png")
    # if fid == 'PS':
    #     return eeg_data, eye_data[:, 1:], train_split, val_split, test_split
    # else:
    return eeg_data, eye_data[:, 1:], train_split, val_split, test_split, train_split_full


# data_reader_wrapper('DDir', 0.75)
