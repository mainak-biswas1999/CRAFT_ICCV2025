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

def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y.ravel(), y_pred.ravel()))

def get_radius(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2)

def get_rmse(y, pred_y):
    #return np.mean(np.sqrt((y[:, 0] - pred_y[:, 0])**2 + (y[:, 1] - pred_y[:, 1])**2))
    return np.linalg.norm(y - pred_y, axis=1).mean()

def get_angle(y, y_pred):
    return np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - y_pred.ravel()), np.cos(y - y_pred.ravel())))))

def plot_predictions(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
    import pingouin as png
    
    y_age = y = np.squeeze(y)
    pred_y_age = pred_y = np.squeeze(pred_y)
    
    #y_age = scale_obj.inv_scale(y)
    #pred_y_age = scale_obj.inv_scale(pred_y)
    
    _min = np.min([np.min(y_age), np.min(pred_y_age)]) 
    _max = np.max([np.max(y_age), np.max(pred_y_age)])
    
    #generate the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.scatter(y_age, pred_y_age, alpha=0.95, s=300, edgecolor='black', linewidth=1)
    plt.xlabel("Actual "+addon, fontsize=45)
    plt.ylabel("Predicted "+addon, fontsize=45)
    
    if _min_use is None and _max_use is None:
        plt.xlim(_min - 1, _max + 1)
        plt.ylim(_min - 1, _max + 1)
    elif _min_use is not None and _max_use is None:
        plt.xlim(_min_use, _max + 1)
        plt.ylim(_min_use, _max + 1)
    elif _min_use is None and _max_use is not None:
        plt.xlim(_min - 1, _max_use)
        plt.ylim(_min - 1, _max_use)
    else:
        plt.xlim(_min_use, _max_use)
        plt.ylim(_min_use, _max_use)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=28)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    ax.tick_params(width=8)
    plt.title(title_, fontsize=35, fontname="Myriad Pro")
    yex = np.linspace(_min - 1, _max + 1, 10000)
    if circ == False:
        corr = png.corr(np.squeeze(y_age), np.squeeze(pred_y_age), method='percbend')
        mse_error = np.round(np.sqrt(np.mean((y_age - pred_y_age)**2)), 2)
        mae_error = np.round(np.mean(np.abs((y_age - pred_y_age))), 2)
        #print(mse_error, mae_error)
        ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}, mae={}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), str(mse_error), str(mae_error)), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=28)
        ################################print best fit#############################################
        A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
        w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
            
        y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
        plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=3, color='red')
    else:
        corr, pval = png.circ_corrcc(np.squeeze(y_age), np.squeeze(pred_y_age), correction_uniform=True)
        mse_error = get_angle(np.squeeze(y_age), np.squeeze(pred_y_age))
        #print(mse_error, mae_error)
        ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}".format(str(np.round(np.abs(corr), 3)), np.maximum(np.round(pval, 3), 0.001), str(np.round(mse_error, 2))), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=28)
    
    ################################print y=x##################################################
    plt.plot(yex, yex, linestyle = 'dashed', linewidth=4, color='black')
    
        
    #plt.title("r= {}, p= {}".format(np.round(corr['r'][0], 2), np.round(corr['p-val'][0], 3)))
    plt.savefig(saveloc)
    plt.tight_layout()
    plt.close()

def plot_curves(loss, __title__, y_label, n_epochs, saveloc, x_label='Epoch', x_axis_vals=[]):
    plt.figure(figsize=(12, 8))
    if len(x_axis_vals) != 0:
        plt.plot(x_axis_vals, loss)
    else:
        plt.plot(np.linspace(0, n_epochs, len(loss)), loss)
    plt.xlabel(x_label, size=35)
    plt.ylabel(y_label, size=35)
    plt.title(__title__, size=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.tick_params(width=8)
    plt.tight_layout()
    plt.savefig(saveloc)


class progressive_mixup(object):

    def __init__(self, type_='r', output_shape=1, alpha=0.01, bs=16, lr=0.0001, n_bins=500):
        self.ymin = 0.0
        self.ymax = 1.0
        self.n_bins = n_bins
        self.alpha = alpha
        self.lr = lr
        self.bs = bs
        self.output_shape = output_shape
        self.model_source = None        #classification based on features - C o F for the source
        self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)

        
        self.type_  = type_

    def plot_histo(self, density_est_x, density_est_y, xlabel_, y_label, title_, saveloc):
        #plot the histogram of velocities
                
        plt.plot(density_est_x, density_est_y, 'r')
        print(auc(density_est_x, density_est_y))


        plt.xlabel(xlabel_, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.title(title_, fontsize=16)
        plt.savefig(saveloc)
        plt.close()
    
    def save_model(self, saveloc):
        self.model_source.save(saveloc)
        
    def load_model(self, saveloc):
        self.model_source = load_model(saveloc, compile=False)
        self.model_source.compile()
        
    def make_model(self, pretrain_loc=None):
        #get the predictor model
        #the feature extractor output
        obj = resnet_model()
        if pretrain_loc is None:
            self.model_source = obj.make_model(to_ret=True)
        else:
            self.model_source = obj.load_model(pretrain_loc, True)
        
        del obj
        self.model_source.compile()
        self.model_source.summary()
    
    def predict(self, X, bs=16):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/bs))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*bs
            
            all_ops.append(self.model_source(X[i*bs : end_pos]).numpy())
        
        return np.concatenate(all_ops)

    def predict_mse(self, X, with_dropout=False):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.bs))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.bs
            #print(i*self.hyperparameter['bs_pred'], end_pos)
            
            age_pred = self.model_source(X[i*self.bs : end_pos], training=with_dropout).numpy()
            all_ops.append(age_pred.squeeze(axis=-1))
            
        return np.concatenate(all_ops)

    def calculate_uncertainty(self, X):
        pred_list = []
        for j in range(10):
            all_pred = np.expand_dims(self.predict_mse(X, with_dropout=True), axis=-1)   #network(feat)
            pred_list.append(all_pred)

        pred_list = np.concatenate(pred_list, axis=-1)  # 20, N, Nbins
        # print(pred_list)
        preds = np.mean(pred_list, axis=1)
        uncertainty = np.var(pred_list, axis=1)
        # print(preds.shape, uncertainty.shape)
        return preds, uncertainty

    def divide_based_on_uncertainty(self, y, h, n_anchors = 1, n_ele = 4):
        #coded for selecting 1 in 4
        #works on the unlabeled data only
        certain_indices = []
        uncertain_indices = []
        sorted_indices = np.argsort(y)
        
        n_iter = int(len(y)/n_ele)
        for i in range(n_iter):
            if i==n_iter-1:
                end_pos = len(y)
            else:
                end_pos = (i+1)*n_ele

            indices_i = sorted_indices[i*n_ele:end_pos]
            #current batch of similar ages
            h_i = h[indices_i]
            pos_min = np.argmin(h_i)
            certain_indices.append(indices_i[pos_min])
            uncertain_indices.append(np.delete(indices_i, pos_min))
        
        return np.array(certain_indices), np.concatenate(uncertain_indices)

    def get_lambda(self, h, tau=50):
        # print(np.mean(h))
        lam = np.clip(np.random.beta(np.sqrt(np.mean(h))*tau, 0.5), 0.05, 0.95)
        return lam
    
    def convex_combination(self, X1, y1, X2, y2, lam, N=None):
        #None works when X1 and X2 have the same size
        if N is None:   
            indices_1 = np.arange(len(X1))
            indices_2 = np.arange(len(X2))    
        else:
            indices_1 = np.concatenate([np.arange(len(X1)), np.random.choice(len(X1), np.max([0, N-len(X1)]))])
            indices_2 = np.concatenate([np.arange(len(X2)), np.random.choice(len(X2), np.max([0, N-len(X2)]))])
        
        np.random.shuffle(indices_1)
        np.random.shuffle(indices_2)
        X = lam*X1[indices_1[0:N]] + (1.-lam)*X2[indices_2[0:N]]
        y = lam*y1[indices_1[0:N]] + (1.-lam)*y2[indices_2[0:N]]
        return X, y

    def loss_cons(self, batch_target_lab, batch_y_target_lab, alpha, to_train=False):
        
        pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
        loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
        
        return alpha*loss_regression_t

    def back_prop(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab):
        with tf.GradientTape() as tape:
            #classification loss
            loss_regression_t = self.loss_cons(batch_target_lab, batch_y_target_lab, 1.0, to_train=True)
            
        grad_loss_regression = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
        self.optimizer_model_s.apply_gradients(zip(grad_loss_regression , self.model_source.trainable_variables))
        #del grad_class

        with tf.GradientTape() as tape:
            #classification loss
            regularizer = self.loss_cons(batch_target_ulab, batch_y_target_ulab, self.alpha, to_train=True)
            
        #update the classifier gradients
        grad_regularizer = tape.gradient(regularizer, self.model_source.trainable_variables)
        self.optimizer_model_s.apply_gradients(zip(grad_regularizer, self.model_source.trainable_variables))
        
        return loss_regression_t, regularizer 

    def get_indices_batch(self, indices, ctr):
        target_ctr = ctr
        start_pos = (target_ctr*self.bs) % len(indices)
        end_pos = ((target_ctr+1)*self.bs) % len(indices)
        
        if len(indices) < self.bs:
            return indices, 0
        elif end_pos < start_pos:
            domain_selector = np.concatenate([indices[start_pos:len(indices)], indices[0:end_pos]])
            target_ctr = 0
        else:
            domain_selector = indices[start_pos:end_pos]
            target_ctr += 1

        return domain_selector, target_ctr

    def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_val_X, target_val_y, num_epochs, printDetails, m_saveloc):
        # print(discrete_distr.shape)
        # sys.exit()
        
        loss_history = {
                        'loss_conf': [],
                        'loss_uncertain': [],
                        'val_rmse': []
                         }
        # train_data_target_t_lab = np.arange(X_target.shape[0])
        # train_data_target_t, last_batch_sz = self.split_2d(y_target, self.bs)
        # ep_length = int(np.ceil(len(X_target)/self.bs))
        certain_ctr = 0
        uncertain_ctr = 0
        label_ctr = 0 
        best_rmse = 1e11
        best_rmse_pos = 0
        
        for i in range(num_epochs):
            pred_ulab, uncertainty = self.calculate_uncertainty(X_target_ulab)
            certain, uncertain = self.divide_based_on_uncertainty(pred_ulab, uncertainty)
            
            X_certain = np.concatenate([X_target_lab, X_target_ulab[certain]])
            y_certain = np.concatenate([y_target_lab, np.expand_dims(pred_ulab[certain], axis=-1)])
            uncertainty_certain = np.concatenate([np.zeros(y_target_lab.shape[0]), uncertainty[certain]])
            
            X_uncertain = X_target_ulab[uncertain]
            y_uncertain = np.expand_dims(pred_ulab[uncertain], axis=-1)
            uncertainty_uncertain = uncertainty[uncertain]

            # print(X_certain.shape, y_certain.shape, X_uncertain.shape, y_uncertain.shape, X_all.shape, y_all.shape)
            # sys.exit()
            lam = self.get_lambda(uncertainty)
            X_conf, y_conf = self.convex_combination(X_certain, y_certain, X_uncertain, y_uncertain, lam, N=X_certain.shape[0] + X_uncertain.shape[0] - X_target_lab.shape[0])
            X_hybrid, y_hybrid = np.concatenate([X_conf, X_target_lab]), np.concatenate([y_conf, y_target_lab]) 
            del X_conf, y_conf
            X_self_certain, y_self_certain = self.convex_combination(X_target_ulab[certain], np.expand_dims(pred_ulab[certain], axis=-1), X_target_ulab[certain], np.expand_dims(pred_ulab[certain], axis=-1), lam)
            X_self_uncertain, y_self_uncertain = self.convex_combination(X_uncertain, y_uncertain, X_uncertain, y_uncertain, lam)
            X_target_lab_comb, y_target_lab_comb = self.convex_combination(X_target_lab, y_target_lab, X_target_lab, y_target_lab, lam)
            X_self, y_self = np.concatenate([X_target_lab_comb, X_self_certain, X_self_uncertain]), np.concatenate([y_target_lab_comb, y_self_certain, y_self_uncertain]) 
            del X_self_certain, y_self_certain, X_self_uncertain, y_self_uncertain, X_target_lab_comb, y_target_lab_comb
            # print(X_hybrid.shape, y_hybrid.shape, X_self.shape, y_self.shape)

            # sys.exit()
            indices_hybrid = np.arange(X_hybrid.shape[0])
            indices_self = np.arange(X_self.shape[0])
            np.random.shuffle(indices_hybrid)
            np.random.shuffle(indices_self)
            n_times = int(np.ceil(X_self.shape[0]/self.bs))

            for j in range(n_times):
                if j == n_times - 1:
                    end_pos = X_self.shape[0]
                else:
                    end_pos = (j+1)*self.bs
                
                batch_X_target_lab = X_hybrid[indices_hybrid[j*self.bs : end_pos]]
                batch_y_target_lab = y_hybrid[indices_hybrid[j*self.bs : end_pos]]
                batch_X_target_ulab = X_self[indices_self[j*self.bs : end_pos]]
                batch_y_target_ulab = y_self[indices_self[j*self.bs : end_pos]]

                loss_conf, loss_uncertain = self.back_prop(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab, batch_y_target_ulab)
                
                # sys.exit()
                #keep track of the loss
                loss_history['loss_conf'].append(loss_conf.numpy())
                loss_history['loss_uncertain'].append(loss_uncertain.numpy())
            
            pred_val_y = self.model_source.predict(target_val_X)
            loss_history['val_rmse'].append(get_rmse1(target_val_y, pred_val_y))
            if (loss_history['val_rmse'][-1] < best_rmse):  #or ((loss_history['val_rmse'][-1] < 1.05*best_rmse) and (i - best_rmse_pos > 0.5*num_epochs)):
                best_rmse_pos = i
                best_rmse = loss_history['val_rmse'][-1]
                self.save_model(m_saveloc)  
            if i%2 == 0:
                #import pdb;pdb.set_trace()
                print("Losses at epoch {}, Loss Certain: {}, Loss Uncertain: {}, val rmse: {}".format(i+1, loss_history['loss_conf'][-2], loss_history['loss_uncertain'][-2], loss_history['val_rmse'][-1]), file=printDetails, flush=True)
                print("Losses at epoch {}, Loss Certain: {}, Loss Uncertain: {}, val rmse: {}".format(i+1, loss_history['loss_conf'][-2], loss_history['loss_uncertain'][-2], loss_history['val_rmse'][-1]))
        return loss_history



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


def run_finetune(target_train_X, target_train_y, target_val_X, target_val_y, m_saveloc, r_saveloc, pretrain_loc, seed=0, perc_label=0.5, alpha=0.1):
    n_epochs = 5
    pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
    np.random.seed(seed)

    target_train_X_lab, target_train_y_lab, target_train_X_ulab, target_train_y_ulab = return_labeled_subets(target_train_X, target_train_y, perc_label) 
    # target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)

    obj_cdist = progressive_mixup(alpha=alpha)
    obj_cdist.make_model(pretrain_loc)
    
    start_time = datetime.now()
    loss_history = obj_cdist.train_model(target_train_X_lab, target_train_y_lab, target_train_X_ulab, target_train_y_ulab, target_val_X, target_val_y, n_epochs, pdetails, m_saveloc)
    end_time = datetime.now()
    print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
    # obj_cdist.save_model(m_saveloc)
    plot_curves(loss_history['loss_conf'], "Confident Loss", "loss", n_epochs, r_saveloc+"/conf_loss.png", x_label='Epoch')
    plot_curves(loss_history['loss_uncertain'], "Uncertain Loss", "loss", n_epochs, r_saveloc+"/uncertain_loss.png", x_label='Epoch')
    plot_curves(loss_history['val_rmse'], "Validation Error", "Error", n_epochs, r_saveloc+"/val_rmse.png", x_label='Epoch')
    
    pdetails.close()
    del target_train_X, target_train_y, target_val_X, target_val_y
    del obj_cdist

def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, m_saveloc, r_saveloc):
    obj_cdist = progressive_mixup()
    obj_cdist.load_model(m_saveloc)
    fptr = open(r_saveloc+"performance_predictions.txt", 'w')

    y_train = rescale_obj.inv_scale(y_train_ood)
    y_val = rescale_obj.inv_scale(y_val_ood)
    y_test = rescale_obj.inv_scale(y_test_ood)
    
    pred_train_y = rescale_obj.inv_scale(obj_cdist.predict(X_train_ood))
    pred_val_y = rescale_obj.inv_scale(obj_cdist.predict(X_val_ood))
    start_time = datetime.now()
    pred_test_y = rescale_obj.inv_scale(obj_cdist.predict(X_test_ood))
    end_time = datetime.now()
    print("--- {} examples: {} minutes ---".format(X_test_ood.shape[0], (end_time - start_time).total_seconds() / 60.0), file=fptr)

    print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(y_train, pred_train_y), get_rmse1(y_test, pred_test_y), get_rmse1(y_val, pred_val_y)), file=fptr)
    fptr.close()
    # plot_predictions_paper(pred_train_y[:, 0], y_train[:, 0], r_saveloc+"train_r_target_paper.png", title_='Train', addon=" # People") #, )
    # plot_predictions_paper(pred_val_y[:, 0], y_val[:, 0], r_saveloc+"val_r_target_paper.png", title_= 'Validation', addon=" # People")
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Attn-Resnet: DataFree', addon=" # People")

    del obj_cdist

if __name__=='__main__':
    import gc
    
    base_mloc = "./models/Mixup/"
    base_rloc = "./results/Mixup/"
    source_loc = "./models/semi_sup/"

    _, _, _, _, _, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people(log_scale=False)
    for i in [2, 5, 10, 20]:
        if not os.path.exists(base_rloc+str(i)):
                os.mkdir(base_rloc+str(i))
        if not os.path.exists(base_mloc+str(i)):
                os.mkdir(base_mloc+str(i))
        for j in range(3):
            if not os.path.exists(base_rloc+str(i)+"/"+str(j)):
                os.mkdir(base_rloc+str(i)+"/"+str(j))
            if not os.path.exists(base_mloc+str(i)+"/"+str(j)):
                os.mkdir(base_mloc+str(i)+"/"+str(j))
            run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", pretrain_loc=source_loc+str(i)+"/"+str(j)+"/", seed=j, perc_label=(0.01*i))
            run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            
            gc.collect()
            tf.keras.backend.clear_session()