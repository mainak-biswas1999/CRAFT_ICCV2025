import os
import glob
import h5py
import numpy as np
from scipy.io import loadmat


def write_req_filenames(loc, write_loc):
    fptr = open(write_loc+'filenames_req.txt', 'w')
    path_dict = []
    all_fnames = glob.glob(loc+"*.mat")
    #all_fnames.sort()
    for matfiles_path in all_fnames:    
        print(matfiles_path[matfiles_path.rindex("/")+1:], file=fptr)
    
    fptr.close()
    #print(path_dict)
    return path_dict

#write_req_filenames("../EEG_decode/code_journal_k_fold/data/afc/", "./")

#loaders --- eye and eeg
def convert_for_seqNets_eye(data, w_size=6, shift=2):
    dshape = data.shape
    n_windows = 1 + int(1.0*(data.shape[1]-w_size)/shift)
    #print(n_windows)
    #exmp, elec, window time, window size
    windowed_data = np.zeros((dshape[0], n_windows, w_size, dshape[2]))
    ctr = 0
    for i in range(0, dshape[1], shift):
        if i+w_size < dshape[1]:
            windowed_data[:, ctr, :, :] = data[:, i:i+w_size, :]
            ctr += 1
    #print(ctr)
    return windowed_data
    
def convert_for_seqNets(data, w_size=6, shift=2):
    dshape = data.shape
    n_windows = 1 + int(1.0*(data.shape[2]-w_size)/shift)
    #print(n_windows)
    #exmp, elec, window time, window size
    windowed_data = np.zeros((dshape[0], dshape[1], n_windows, w_size))
    ctr = 0
    for i in range(0, dshape[2], shift):
        if i+w_size < dshape[2]:
            windowed_data[:, :, ctr, :] = data[:, :, i:i+w_size]
            ctr += 1
    #print(ctr)
    return windowed_data
    
def get_path_list(loc, ignore_list):
    path_dict = []
    all_fnames = glob.glob(loc+"*.mat")
    all_fnames.sort()
    for matfiles_path in all_fnames:
        
        if matfiles_path[matfiles_path.rindex("/")+1:] in ignore_list:
            continue
        path_dict.append(matfiles_path)
    #print(path_dict)
    return path_dict

def get_eeg_mat(path_dict, ignore_list):
    name_matfiles = get_path_list(path_dict, ignore_list)
    all_trials = []
    block_ids = []
    for file_name in name_matfiles:
        print("Reading: ", file_name[file_name.rindex("/")+1:])
        #read the files
        File_ = h5py.File(file_name)
        #ref_struct = File_['dat_struct'] - File has dat_struct - which has behaviour and eeg_dat
        ref_struct = File_['dat_struct']
        sub_id = int(file_name[-6:-4])
        if sub_id != 9:
            all_trials.append(np.transpose(File_[ref_struct['eeg_dat'][0, 0]], [0, 2, 1])[:, :, 0:414])
            block_ids.append(np.ones(all_trials[-1].shape[0])*(sub_id-1)*2 + 1)
        
        all_trials.append(np.transpose(File_[ref_struct['eeg_dat'][1, 0]], [0, 2, 1])[:, :, 0:414])
        block_ids.append(np.ones(all_trials[-1].shape[0])*sub_id*2)
        
    #is_corr = np.concatenate(is_corr)
    #print(is_corr.shape)
    all_trials = np.concatenate(all_trials)
    
    return all_trials, np.concatenate(block_ids).astype('int')

def load_data(path_dict, ignore_list, stratified=False, kfold=None):
    np.random.seed(24)
    all_trials, block_id = get_eeg_mat(path_dict, ignore_list)
    
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
    return all_trials, train_list, test_list


class Eye_data:
    def __init__(self):
        pass
    
    def read_data(self, path):
        subjects = []
        participants = sorted(os.listdir(path))
        for subject in participants:
            # print(subject)
            if os.path.isdir(os.path.join(path, subject)):
                block1 = []
                block2 = []
                
                block1_files = sorted(os.listdir(os.path.join(path, subject, 'AFC_block1')))
                for file in block1_files:
                    # print("block1 ",file)
                    if file.endswith('.mat'):
                        file_path = os.path.join(path, subject, 'AFC_block1', file)
                        mat = loadmat(file_path)
                        block1.append(mat)
    
                block2_files = sorted(os.listdir(os.path.join(path, subject, 'AFC_block2')))
                for file in block2_files:
                    if file.endswith('.mat'):
                        # print("block2 ",file)
                        file_path = os.path.join(path, subject, 'AFC_block2', file)
                        mat = loadmat(file_path)
                        block2.append(mat)
                subjects.append((block1, block2))   
    
        return np.array(subjects)
        
    def merge_eye_files(self, path):
        
        data = self.read_data(path)
        subjects = []
        for subject in data:
            one_subject = []
            for block in subject:
                blocks = []
                for file in block:
                    blocks.append(file['trial_eye'].T)
                one_subject.append(blocks)
            subjects.append(one_subject)
        trial_wise_data = np.array(subjects).reshape(21, 2, -1)
        return trial_wise_data
    
    def get_rejected_info(self, path):
        mat = loadmat(path)
        toRemove = np.array(mat['toRemove'].T.reshape(-1))
    
        mask = np.ones(400)
        mask[toRemove[4][0][1] - 1] = 0
    
        toTake5 = np.concatenate((np.arange(51, 151), np.arange(201, 501)))[mask == 1]
        mask = np.zeros(500)
        mask[toTake5 - 1] = 1
    
        toRemove[0] = [[[], []]]
        new_toRemove = toRemove
        # new_toRemove[4][0][1] = np.append(np.concatenate((np.arange(1, 51), np.arange(151, 201))), new_toRemove[4][0][1].reshape(-1))
        new_toRemove[0][0][0] = np.concatenate((np.arange(151, 201), np.arange(301, 351)))
        new_toRemove[0][0][1] = np.arange(151, 201)
    
        trials_tokeep = []
        for subject in new_toRemove:
            try:
                task = []
                for block in subject[0]:
                    index_mask = np.ones(500)
                    index_mask[block.reshape(-1) - 1] = 0
                    task.append(index_mask)
                trials_tokeep.append(task)
            except:
                index_mask = np.ones(500)
                trials_tokeep.append([index_mask, index_mask])
    
        trials_tokeep[4][1] = mask
        return np.array(trials_tokeep)
    
    def custom_key(self, string):
        string = string.split('.')[0]
        parts = string.split('_')
        subject_number = int(parts[1])
        block_number = int(parts[3])
        return (subject_number, block_number)
    
    def vectorize_eye_data(self, data_list, len_eye):
        
        eye_data = []
        #blocks
        for i in range(len(data_list)):
            block_eye_data =  data_list[i]
            #trials
            for j in range(len(block_eye_data)):
                eye_data_per_trial = block_eye_data[j]
                len_eye_avail = eye_data_per_trial.shape[0]
                
                if len_eye_avail >= len_eye:
                    eye_data_per_trial = eye_data_per_trial[0:len_eye]
                else:
                    #eye_data_per_trial = np.pad(eye_data_per_trial, ((0, len_eye - len_eye_avail), (0, 0)), 'constant', constant_values=-99999.0)
                    eye_data_per_trial = np.pad(eye_data_per_trial, ((0, len_eye - len_eye_avail), (0, 0)), 'constant', constant_values=np.nan)
                
                
                #print(eye_data_per_trial.shape)
                eye_data.append([eye_data_per_trial])
        
        eye_data = np.concatenate(eye_data)
        #eye_data[np.where(np.isnan(eye_data))] = -99999.0
        #print(eye_data.shape)
        return eye_data
    
    def merge(self, data_folder, eyepath, maskpath, time=1656):
        data_list = []
        name_files = open(data_folder, "r") 
        
        #files = sorted(os.listdir(data_folder), key=self.custom_key)
        files = sorted(name_files.read()[0:-1].split("\n"), key=self.custom_key)
        
        #print(files, files2)
        #os.exit()
        
        #labels_array = self.merge_eye_files()
        labels_array = self.merge_eye_files(eyepath)
        #print(labels_array.shape)
        #labels_mask  = self.get_rejected_info)
        labels_mask  = self.get_rejected_info(maskpath)
        
        for filename in files:
            #print(filename)
            if filename.endswith('.mat'):
                # Extract subject ID and block ID from the filename
                filename = filename.split('.')[0]
                parts = filename.split('_')
                subject_id = int(parts[1])
                block_id = int(parts[3])
    
                # Find corresponding labels
                mask = labels_mask[subject_id - 1, block_id - 1]
                labels = labels_array[subject_id - 1, block_id - 1][mask == 1]
                # labels = labels_array[subject_id - 1, block_id - 1]
                data_list.append(labels)
                
        #data_list = np.array(data_list, dtype='object')
        #print(data_list.shape)
        return self.vectorize_eye_data(data_list, time)
        

#prep data
def gaussian_smoothing_eye(data):
    #of the dimension - (N, t, w, 2)
    dshape = data.shape
    #dshape = (16843, 205, 24, 2)
    filter = np.arange(dshape[2])
    mu = np.mean(filter)
    sd = np.std(filter)
    filter = np.exp(-1.0*(filter - mu)**2/(2 * sd**2))
    #print(mu, sd)
    filter = filter.reshape((1, 1, dshape[2], 1))
    #print(filter.shape)
    
    filter_norm = filter/np.sum(filter)
    #print(filter_norm, np.sum(filter_norm))
    return np.nansum(data * filter_norm, axis=2)

 
def data_load_driver():
    obj_eye = Eye_data()
    eye_data = obj_eye.merge('./data/filenames_req.txt', './data/Preprocessed_1.00/', './data/toRemove_noSacc_eyeLoss_1.00_specSubIncl.mat')
    #nan_places = np.where(np.isnan(eye_data))
    #print(nan_places[0].shape, nan_places[0].shape, nan_places[2].shape, np.nanmin(eye_data), np.nanmax(eye_data))
    eye_data = convert_for_seqNets_eye(eye_data, w_size=24, shift=8)
    print("Eye tracking data shape: ", eye_data.shape)
    
    eye_data = gaussian_smoothing_eye(eye_data)
    print("Eye tracking data after exp_weighting: ", eye_data.shape)
    ##print(data_list[1000])
    #
    #all_trial_eeg = np.transpose(load_data("./data/SricharanRAW/", []), [0, 2, 1, 3])
    #will do a stratified k-fold loading for each participants
    #all_trial_eeg = np.expand_dims(load_data("./data/SricharanRAW/", []), axis=-1)
    all_trial_eeg, train_list, test_list = load_data("./data/SricharanRAW/", [], stratified=True, kfold=5)
    all_trial_eeg = np.expand_dims(all_trial_eeg, axis=-1)
    print("eeg data shape: ", all_trial_eeg.shape)
    
    return all_trial_eeg, eye_data, train_list, test_list

#a, b, c, d = data_load_driver()
