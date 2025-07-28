import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from read_data import * 
from deepak_eye_dat_prep import *
import sys


def resample_freq(eeg_data, old_freq, new_freq):
    t_total = eeg_data.shape[2] * 1./old_freq
    
    xnew = np.arange(0, t_total, 1./new_freq)
    xold = np.arange(0, t_total, 1./old_freq)
    
    eeg_data_new = np.zeros((eeg_data.shape[0], eeg_data.shape[1], xnew.shape[0]), dtype='float32')
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            eeg_data_new[i, j, :] = np.interp(xnew, xold, eeg_data[i, j, :])
    return eeg_data_new


def eye_tracking(basepath, max_id=33, n_blocks=8):
    eye_tracking = []
    for i in range(1, max_id):
        #print("Reading Eye: Sub", i+1)
        dirname = basepath+"Sub"+str(i+1)+"_RawET_table/"
        if os.path.exists(dirname):
            for j in range(n_blocks):
                fname = dirname+"WME_ET_rawTable_S"+str(i+1)+"_B"+str(j+1)+".csv"
                if os.path.exists(fname):
                    sub_edat = pd.read_csv(fname, usecols=["Trial", "LPORX_dva", "RPORY_dva"]).to_numpy()
                    trial_ids = np.unique(sub_edat[:, 0]).astype('int')
                    if trial_ids.shape[0] < 125:
                       print("Sub {}, Block {}, Ntrials {}".format(i+1, j+1, trial_ids.shape[0]))
                       continue
                        
                    for t_no in trial_ids:
                        index = np.searchsorted(sub_edat[:, 0], t_no)
                        eye_tracking.append([sub_edat[index:index+1050, 1:]])
                else:
                    print(fname, " does not exist")
    return np.concatenate(eye_tracking)

def get_eeg_WM(basepath, max_id=33, t_start=496, t_max=1008):
    sub_ids = []
    all_trials = []
    for i in range(1, max_id):
        dirname = basepath+"Sub"+str(i+1).zfill(2)
        if os.path.exists(dirname+"/10_512fs_EEG.mat"):
            print("Reading EEG: ", dirname[-5:])
            try:
                File_ = loadmat(dirname+"/10_512fs_EEG.mat")
                temp_dat = np.transpose(File_['data'], [2, 0, 1])
            except:
                File_ = h5py.File(dirname+"/10_512fs_EEG.mat")
                temp_dat = np.transpose(File_['data'], [0, 2, 1])
            #print(temp_dat.shape)
            sub_id = int(dirname[-2:])
            #some hard coding for missing eeg data
            if (i+1) == 15: #block 3, 4 missing
                #print(temp_dat.shape)
                all_trials.append(temp_dat[0:250, :, t_start:t_max])
                all_trials.append(temp_dat[500:1000, :, t_start:t_max])

                sub_ids.append(np.ones(all_trials[-1].shape[0] + all_trials[-2].shape[0])*(sub_id-1))
            elif (i+1)==24: #block 2, 4 missing
                all_trials.append(temp_dat[0:125, :, t_start:t_max])
                all_trials.append(temp_dat[375:, :, t_start:t_max])
                sub_ids.append(np.ones(all_trials[-1].shape[0] + all_trials[-2].shape[0])*(sub_id-1))
            elif (i+1) == 13:   #Block 8 missing
                all_trials.append(temp_dat[0:875, :, t_start:t_max])
                sub_ids.append(np.ones(all_trials[-1].shape[0])*(sub_id-1))
            elif (i+1) == 29: #block 29, 2 missing
                all_trials.append(temp_dat[0:125, :, t_start:t_max])
                all_trials.append(temp_dat[250:, :, t_start:t_max])
                sub_ids.append(np.ones(all_trials[-1].shape[0] + all_trials[-2].shape[0])*(sub_id-1))
            else:
                all_trials.append(temp_dat[:, :, t_start:t_max])
                sub_ids.append(np.ones(all_trials[-1].shape[0])*(sub_id-1))
            # print(all_trials[-1].shape)
        
    
    all_trials = np.concatenate(all_trials)
    
    return all_trials, np.concatenate(sub_ids).astype('int')

def load_data_WM(path_dict, max_id=33, stratified=False, kfold=None, load_data=True):
    np.random.seed(24)
    if load_data == True:
        all_trials, block_id = get_eeg_WM(path_dict, max_id)
    else:
        all_trials = None
        block_id = path_dict
    
    if stratified == False:
        return all_trials
        
    
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
    return all_trials, train_list, test_list, block_id
    

def rem_from_list(train_list, test_list, to_remlist):
    for i in range(len(train_list)):
        for rem_ele in to_remlist:
            if rem_ele in train_list[i]:
                train_list[i].remove(rem_ele)
            elif rem_ele in test_list[i]:
                test_list[i].remove(rem_ele)
        #print(len(train_list[i]), len(test_list[i]))
    #sys.exit()

    return train_list, test_list


def create_data_npz_file():
    eye_proc_obj = prepEyeTracking()
    pos_eye_data = eye_proc_obj.conv_to_pix(read_eye_tracking_WM("./data_deepak/eye-tracking/"))
    pos_eye_data = eye_proc_obj.gaussian_smoothing_eye(pos_eye_data)


    #pos_eye_data = eye_proc_obj.rem_nans(pos_eye_data)
    #time dt -> 2 ms, c
    acc_eye_data = eye_proc_obj.magnitude_of_acc_from_pos(pos_eye_data, 0.002)
    speed_eye_data = eye_proc_obj.magnitude_of_vel_from_pos(pos_eye_data, 0.002)
    saccade_onsets = eye_proc_obj.get_saccade_onset(speed_eye_data, 685, -10, 150)
    
    to_remlist1 = eye_proc_obj.rem_blinks(acc_eye_data, 14.5, 6)
    eye_data, to_remlist2 = eye_proc_obj.displacement_vec(pos_eye_data, saccade_onsets, s=38, w=50)

    to_remlist = list(set(to_remlist1)|set(to_remlist2))

    print("Numbe of wrong displacement calculaion:, ", len(to_remlist))
    
    all_trial_eeg, _, _, block_id = load_data_WM("./data_deepak/EEG/", 33, stratified=True, kfold=5)
    
    
    to_remlist = list(set(np.where(np.isnan(all_trial_eeg))[0])|set(to_remlist))
    #train_list, test_list = rem_from_list(train_list, test_list, to_remlist)

    all_trial_eeg = resample_freq(all_trial_eeg, 512, 500)
    all_trial_eeg = np.expand_dims(all_trial_eeg, axis=-1)
    
    all_trial_eeg = np.delete(all_trial_eeg, to_remlist, axis=0)
    eye_data = np.delete(eye_data, to_remlist, axis=0)
    block_id = np.delete(block_id, to_remlist, axis=0)

    print("eeg data shape: ", all_trial_eeg.shape, block_id.shape)
    print("Eye tracking data after exp_weighting: ", eye_data.shape)
    print(np.where(np.isnan(all_trial_eeg)))
    
    del_x = eye_data[:, 0]
    del_y = eye_data[:, 1]
    
    print(np.min(del_x), np.max(del_x), np.min(del_y), np.max(del_y))
    #erroneous saccades calculations - by construction they can't exist
    rem_x = np.where((del_x > 750) + (del_x < -750))[0]

    rem_y = np.where((del_y > 100) + (del_y < -100))[0]
    print(len(rem_x), len(rem_y))
    to_remlist = list(set(rem_x)|set(rem_y))
    print(len(to_remlist))
    all_trial_eeg = np.delete(all_trial_eeg, to_remlist, axis=0)
    eye_data = np.delete(eye_data, to_remlist, axis=0)
    block_id = np.delete(block_id, to_remlist, axis=0)
    np.savez('./data_deepak/WM_data_full.npz', eeg_data=all_trial_eeg, eye_data=eye_data.astype('float32'), block_id=block_id.astype('int32'))
    #return all_trial_eeg, eye_data, train_list, test_list

def leave_sub_out(block_id, perc_data=1.0):
    np.random.seed(24)
    train_list, test_list = [], []
    parts = np.unique(block_id) #--- 22
    n_parts = len(parts)
    # print(len(np.unique(block_id)))     #-- 22
    # selected ordering - 4, 4, 5, 5, test - 4
    test_list_ids = np.random.choice(np.arange(n_parts), 4, replace=False)
    train_list_ids = np.delete(np.arange(n_parts), test_list_ids)
    
    val_list_selector = np.random.choice(np.arange(len(train_list_ids)), 4, replace=False)
    val_list_ids = train_list_ids[val_list_selector]

    train_split_full = np.delete(train_list_ids, val_list_selector)
    train_list_ids = train_split_full[0:int(train_split_full.shape[0]*perc_data)]
    # print(test_list_ids, train_list_ids, val_list_ids)
    
    test_list_ids = list(parts[test_list_ids])
    train_list_ids = list(parts[train_list_ids])
    val_list_ids = list(parts[val_list_ids])
    train_list_full_ids = list(parts[train_split_full])

    train_list = [tno for (tno, index) in enumerate(block_id) if index in train_list_ids]
    test_list = [tno for (tno, index) in enumerate(block_id) if index in test_list_ids]
    val_list = [tno for (tno, index) in enumerate(block_id) if index in val_list_ids]

    train_list_full = [tno for (tno, index) in enumerate(block_id) if index in train_list_full_ids]  

    print(len(train_list_full), len(train_list), len(test_list), len(val_list))
    return train_list, test_list, val_list, train_list_full

def convert_to_polar(eye):
    #print(eye.shape)
    no_saccade = []
    r = np.expand_dims(np.sqrt(np.sum(eye**2, axis=1)), axis=-1)
    # no_saccade = np.where(r<50)[0]
    #it is the angle of the displacement vector  - y first and x second
    theta = np.expand_dims(np.arctan2(eye[:,1], eye[:,0]), axis=-1)

    data = np.concatenate([r, theta], axis=1)
    data = np.delete(data, no_saccade, axis=0)
    
    # eye_proc_obj = prepEyeTracking()
    # eye_proc_obj.plot_velocity_hist(data[:, 0], False, 100, "r (in pix)", "Density", "del r", "./Results/pilot_models/stats/del_r_rem50.png")
    # eye_proc_obj.plot_velocity_hist(data[:, 1], False, 100, "theta (in radians)", "Density", "del theta", "./Results/pilot_models/stats/del_theta_rem50.png")

    return data, no_saccade

#sub id is ame as block id here
def data_loader_WM(perc_data=1.0):
    full_file = np.load('./data_deepak/WM_data_full.npz')

    eye_data, no_saccade = convert_to_polar(full_file['eye_data'])
    
    all_trial_eeg = np.delete(full_file['eeg_data'], no_saccade, axis=0)
    block_id = np.delete(full_file['block_id'], no_saccade, axis=0)
    
    # print(all_trial_eeg.shape, block_id.shape, eye_data.shape)
    #print(len(train_list[0]), len(test_list[0]), list(set(test_list[0]) & set(test_list[1])))
    #print(np.min(eye_data[:, 0]), np.max(eye_data[:, 0]),np.min(eye_data[:, 1]), np.max(eye_data[:, 1]))
    #print(all_trial_eeg.shape, eye_data.shape, block_id.shape, all_trial_eeg.dtype, eye_data.dtype, block_id.dtype)

    train_list, test_list, val_list, train_split_full = leave_sub_out(block_id, perc_data)

    return all_trial_eeg, eye_data, train_list, val_list, test_list, train_split_full

#create_data_npz_file()
# data_loader_WM(1.0)
# data_loader_WM(0.25)