from plotter import *
from read_data import *
from resnet import *
import os
import sys
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.metrics import auc
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda, UpSampling1D, BatchNormalization, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
from datetime import datetime
import math
from scipy.stats import norm

gpus = tf.config.experimental.list_physical_devices('GPU')


if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=35000)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)    

matplotlib.use('Agg') 



def cal_cdf(x, mean, std):
    return norm.cdf((x-mean)/std)

def cal_block_indices(mean, std, side_info):
    """
    Args:
        mean: mean of the prediction
        std: std of the predicton
        side_info: a tuple giving the information of density map, (min, max, number_of_block+1, block_size)
    Returns:
        a list of (slot, density)
    """
    den_list = []
    min_3sigma = mean - 3*std
    max_3sigma = mean + 3*std
    partitions = np.linspace(*side_info[:3])
    in_range = []
    for p in partitions:
        if min_3sigma < p < max_3sigma:
            in_range.append(p.item())
    if in_range:
        for i in range(len(in_range)):
            slot = round((in_range[i] - side_info[3] - side_info[0]) / side_info[3])
            if i == 0:
                den = cal_cdf(in_range[i], mean, std)
            else:
                den = cal_cdf(in_range[i], mean, std) - cal_cdf(in_range[i-1], mean, std)
            den_list.append([slot, den])
        if in_range[-1] != partitions[-1]:
            den_list.append([round((in_range[-1] - side_info[0]) / side_info[3]), 1 - cal_cdf(in_range[-1], mean, std)])
    else:
        for p in partitions:
            if min_3sigma <= p:
                slot = round((p - side_info[3] - side_info[0]) / side_info[3])
                den_list.append([slot, 1])
                break
    return den_list

def get_outputs(model, X, with_dropout=False, bs=16):
    all_ops = []
    n_times = int(np.ceil(X.shape[0]/bs))
        
    for i in range(n_times):
        if i == n_times - 1:
            end_pos = X.shape[0]
        else:
            end_pos = (i+1)*bs
        
        all_ops.append(model(X[i*bs : end_pos], training=with_dropout).numpy())
    
    return np.concatenate(all_ops)

def calculate_threshold(x, p):
    x=np.squeeze(x)
    x_sorted = np.sort(x)
    index = int(np.ceil(p*len(x)))
    return x_sorted[index]

def calculate_q_func(model_loc, X, y):
    n_q = int(np.min([20, np.ceil(0.1*len(y))]))  #no of datapoints to create to fit q
    model = load_model(model_loc)
    model.compile()
    q = (1, 0)
    
    pred_list = []
    for j in range(10):
        all_pred = np.squeeze(get_outputs(model, X, with_dropout=True))   #network(feat)
        pred_list.append([all_pred])

    pred_list = np.concatenate(pred_list).T
    var = np.squeeze(np.var(pred_list, axis=1))
    # print(pred_list.shape)
    # collecting prediction
    outputs = np.squeeze(get_outputs(model, X, with_dropout=False))
    errors = np.abs(outputs - y.ravel())
    # print(var.shape, outputs.shape, errors.shape)
    var_sort = np.argsort(var)
    bin_sz = int(len(y)/n_q) 
    
    u_s = []
    e_s = []
    for i in range(n_q):
        if i == n_q - 1:
            end_pos = len(y)
        else:
            end_pos = (i+1)*bin_sz
        
        indices = var_sort[i*bin_sz : end_pos]
        u_s.append(np.mean(var[indices]))
        e_s.append(np.std(errors[indices]))

    u_s, e_s = np.array(u_s), np.array(e_s)
    # print(u_s.shape, e_s.shape)
    u_s = np.concatenate([np.expand_dims(u_s, axis=-1), np.ones((n_q, 1))], axis=-1)
    e_s = np.expand_dims(e_s, axis=-1)
    # print(u_s.shape, e_s.shape)
    # print(u_s)
    # print(e_s)
    clf = Ridge(alpha=0.1, fit_intercept=False)
    clf.fit(u_s, e_s)
    # print(clf.coef_, clf.coef_.shape)
    #dim 0 is slope
    del model
    return (clf.coef_[0, 0], clf.coef_[0, 1])

# def gen_pseudo_label(model_path, data_path, q_func, block_size, device):
def TASFAR_gen_pseudo_label(model, target_train_X, q_func):
    # Given parameters
    THRESHOLD_percentile = 0.25 # again the source data is unknown and hence we can do a percentile split to define the threshold (25 % of the data is confident) 
    #since the source data is unavailable to us, computing the Q function is not trivial -- hence we use the variace directly
    block_size = 1.0/500.0  #keeping it same as craft
    eps = 1e-9 #for numerical stability
    """
    Args:
        model_path: path of pretrained model
        data_path: data path of target domain
        q_func: q function
        block_size: block size of density map
        device: device
    Returns:
        pseudo_label_dict: a dictionary containing the pseudo label
    """
    slope, intercept = q_func
    pred_data = []  # [x, y, var, frame_id]
    pred_list = []
    for j in range(10):
        all_pred = np.squeeze(get_outputs(model, target_train_X, with_dropout=True))   #network(feat)
        pred_list.append([all_pred])

    pred_list = np.concatenate(pred_list).T
    var = np.var(pred_list, axis=1, keepdims=True)
    # print(pred_list.shape)
    # collecting prediction
    pred = get_outputs(model, target_train_X, with_dropout=False)
    # pred_data.append([prediction, slope*var+intercept, var, data_index.item(), label.item()])

    # Generate density map
    pred_data = np.concatenate([pred, slope*var+intercept, var], axis=1)
    # print(pred_data.shape)
    
    #for advantage -- since the y are scaled between -1 and 1
    # min_data = pred_data[:, 0] - 3 * pred_data[:, 1]
    # max_data = pred_data[:, 0] + 3 * pred_data[:, 1]
    minimum = 0.0
    maximum = 1.0
    num_block = math.ceil((maximum - minimum) / block_size)
    side_info = (minimum, minimum+num_block*block_size, num_block+1, block_size)

    THRESHOLD = calculate_threshold(pred_data[:, 2], THRESHOLD_percentile)
    # Generate density map
    den_map = np.zeros(num_block)
    for data in pred_data:
        if data[2] < THRESHOLD:
            den_list = cal_block_indices(data[0], data[1], side_info)
            for d in den_list:
                den_map[d[0]] += d[1]
    den_map /= (np.sum(den_map) + 1e-9)

    # Generate estimation map
    est_map = (np.linspace(*side_info[:3]) * 2 + block_size) / 2
    # print(den_map)
    # Generate pseudo label dictionary
    # pseudo_label_dict = {}
    #just need the pseudo-label
    pseudo_label_list = []
    for data in pred_data:
        if data[2] <= THRESHOLD:
            pseudo_label_list.append(data[0])
            # pseudo_label_dict[int(data[3])] = {
            #     'pseudo_label': data[0],
            #     'variance': data[2],
            #     'lmd': 1/den_map.shape[0],
            #     'gmd': 1/den_map.shape[0],
            #     'label': data[4],
            #     'prediction': data[0]
            # }
        else:
            den_list = cal_block_indices(data[0], data[1], side_info)
            pseudo_list = []  # To be used for interpolation
            for d in den_list:
                pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
            pseudo_array = np.array(pseudo_list)
            pseudo_label = np.average(pseudo_array[:, 0], weights=(pseudo_array[:, 1] + eps)).item()  
            pseudo_label_list.append(pseudo_label)
            # lmd = np.mean(pseudo_array[:, 2]).item()
            # pseudo_label_dict[int(data[3])] = {
            #     'pseudo_label': pseudo_label,
            #     'variance': data[2],
            #     'lmd': lmd,
            #     'gmd': 1 / den_map.shape[0],
            #     'label': data[4],
            #     'prediction': data[0]
            # }
    pseudo_label_list = np.expand_dims(np.array(pseudo_label_list), axis=-1)
    # return pseudo_label_dict
    return pseudo_label_list


def return_labeled_subets(X, y, perc_label):
    #stratified labelling in groups of 100s
    y_sort = np.argsort(np.squeeze(y))
    n_per_split = 100
    n_bins = int(np.ceil(len(y)/n_per_split))
    #print(n_bins)
    n_to_select = n_per_split*perc_label
    lab = []
    ulab = []
    for i in range(n_bins):
        if i<n_bins-1:
            st, en = i*n_per_split, (i+1)*n_per_split
        else:
            st, en = i*n_per_split, len(y)

        indices = y_sort[st:en]

        n_lab = int(perc_label*len(indices))
        
        lab_indices = np.random.choice(len(indices), n_lab, replace=False)
        lab.append(indices[lab_indices])
        
        indices = np.delete(indices, lab_indices)
        ulab.append(indices)

    lab = np.concatenate(lab)
    ulab = np.concatenate(ulab)

    return X[lab], y[lab], X[ulab], y[ulab] 

def run_finetune(target_train_X, target_train_y, target_val_X, target_val_y, m_saveloc, r_saveloc, model_Q=None, pretrain_loc=None, seed=0, perc_label=0.5, alpha=0.1):
    n_epochs = 10
    
    pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
    
    np.random.seed(seed)

    target_train_X_lab, target_train_y_lab, target_train_X_ulab, _ = return_labeled_subets(target_train_X, target_train_y, perc_label) 
    del target_train_X, target_train_y
    
    # start_time = datetime.now()
    
    if pretrain_loc is None:
        print("Error: TASFAR only works for this case, else will not give any results -- absolutely need a pretrained model")
        sys.exit()
    
    if model_Q is None:
        q_func = (1, 0)
    else:
        # estimate q, based on the small amount of semisupervised labels
        q_func = calculate_q_func(model_Q, target_train_X_lab, target_train_y_lab)
    # print(q_func)
    # return
    
    model_pseduo = load_model(pretrain_loc)
    model_pseduo.compile()
    model_pseduo.summary()
    pseudo_labels_tasfar = TASFAR_gen_pseudo_label(model_pseduo, target_train_X_ulab, q_func)
    
    target_train_X = np.concatenate([target_train_X_lab, target_train_X_ulab], axis=0)
    target_train_y = np.concatenate([target_train_y_lab, pseudo_labels_tasfar], axis=0)
    weights = np.concatenate([np.ones((len(target_train_y_lab), 1)), alpha*np.ones((len(pseudo_labels_tasfar), 1))], axis=0)

    del model_pseduo, target_train_X_lab, target_train_X_ulab, target_train_y_lab, pseudo_labels_tasfar
    print(target_train_X.shape, target_train_y.shape)

    obj = resnet_model(type_='r_weighted')
    obj.load_model_weighted_outs(pretrain_loc)
    
    start_time = datetime.now()
    obj.train_model(target_train_X, target_train_y, m_saveloc, r_saveloc, num_epochs=n_epochs, val_data=target_val_X, val_label=target_val_y, weights=weights)
    end_time = datetime.now()

    print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
    # obj_cdist.save_model(m_saveloc)
    pdetails.close()
    del target_train_X, target_train_y
    del obj


def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, m_saveloc, r_saveloc):
    obj_cdist = resnet_model(type_='r_weighted')
    obj_cdist.load_model(m_saveloc)

    fptr = open(r_saveloc+"performance_predictions.txt", 'w')

    y_train = y_train_ood
    y_val = y_val_ood
    y_test = y_test_ood
    
    pred_train_y = obj_cdist.predict(X_train_ood)
    pred_val_y = obj_cdist.predict(X_val_ood)
    start_time = datetime.now()
    pred_test_y = obj_cdist.predict(X_test_ood)
    end_time = datetime.now()
    print("--- {} examples: {} minutes ---".format(X_test_ood.shape[0], (end_time - start_time).total_seconds() / 60.0), file=fptr)

    print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(y_train, pred_train_y), get_rmse1(y_test, pred_test_y), get_rmse1(y_val, pred_val_y)), file=fptr)
    fptr.close()
    # plot_predictions_paper(pred_train_y[:, 0], y_train[:, 0], r_saveloc+"train_r_target_paper.png", title_='Train', addon=" # People") #, )
    # plot_predictions_paper(pred_val_y[:, 0], y_val[:, 0], r_saveloc+"val_r_target_paper.png", title_= 'Validation', addon=" # People")
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Attn-Resnet: TASFAR', addon=" # People")

    del obj_cdist



if __name__=='__main__':
    import gc
    
    base_mloc = "./models/TASFAR/"
    base_rloc = "./results/TASFAR/"
    source_loc = "./models/semi_sup/"
    X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood = read_cam('/data1/mainak/cancer_pred/data/cam17.npz')
    for i in [1, 2, 5, 10]:
        if not os.path.exists(base_rloc+str(i)):
                os.mkdir(base_rloc+str(i))
        if not os.path.exists(base_mloc+str(i)):
                os.mkdir(base_mloc+str(i))
        for j in range(3):
            if not os.path.exists(base_rloc+str(i)+"/"+str(j)):
                os.mkdir(base_rloc+str(i)+"/"+str(j))
            if not os.path.exists(base_mloc+str(i)+"/"+str(j)):
                os.mkdir(base_mloc+str(i)+"/"+str(j))
            
            run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", model_Q=source_loc+str(i)+"/"+str(j)+"/", pretrain_loc=source_loc+str(i)+"/"+str(j)+"/", seed=j, perc_label=(0.01*i))
            run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            
            gc.collect()
            tf.keras.backend.clear_session()