from scipy.stats import norm
import os
import sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from datetime import datetime
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from sfcn_tensorflow import *
from loader import *
from data_struct import *
from plotter import *
from plot_paper import *
import matplotlib
import math

matplotlib.use('Agg')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#       tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=38000)]) # Notice here
#       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#       # Virtual devices must be set before GPUs have been initialized
#       print(e)

class progressive_mixup(object):
    def __init__(self, output_shape=1, alpha=0.1, bs=4, lr=0.0001, n_bins = 400):
        self.ymin = 0.0
        self.ymax = 1.0
        self.n_bins = n_bins
        self.alpha = alpha
        self.lr = lr
        self.bs = bs
        self.output_shape = output_shape
        self.model_source = None        #classification based on features - C o F for the source
        self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
    
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
        
    def make_model(self):
        #get the predictor model
        #the feature extractor output
        mod_object = SFCN_tf(pretrained=True, to_train_full=True)
        self.model_source = mod_object.add_mse_head(True)
        self.model_source.compile()
        self.model_source.summary()
        del mod_object
        
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

    def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_train_y, num_epochs, printDetails, m_saveloc):
        # print(discrete_distr.shape)
        # sys.exit()
        
        loss_history = {
                        'loss_conf': [],
                        'loss_uncertain': [],
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
            
            if i%2 == 0:
                #import pdb;pdb.set_trace()
                print("Losses at epoch {}, Loss Certain: {}, Loss Uncertain: {}".format(i+1, loss_history['loss_conf'][-2], loss_history['loss_uncertain'][-2]), file=printDetails, flush=True)
                print("Losses at epoch {}, Loss Certain: {}, Loss Uncertain: {}".format(i+1, loss_history['loss_conf'][-2], loss_history['loss_uncertain'][-2]))
        return loss_history

def return_labeled_subets(X, y, perc_label):
    #stratified labelling in groups of 100s
    y_sort = np.argsort(np.squeeze(y))
    # y_sort = np.argsort(np.squeeze(y))
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

def caller_paper_subsamp_kseed(base_rloc_r, base_mloc_r, perc_label=0.2, alpha=0.1, check_train=False):
    bin_size = 1
    n_epochs = 25    #keep the no of updates fixed
    n_fold = 4
    age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"
    if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
       os.mkdir(base_rloc_r+str(int(perc_label*100)))
    if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
       os.mkdir(base_mloc_r+str(int(perc_label*100)))
    base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
    base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"
    # if not os.path.exists(base_rloc_r+str(alpha)):
    #     os.mkdir(base_rloc_r+str(alpha))
    # if not os.path.exists(base_mloc_r+str(alpha)):
    #     os.mkdir(base_mloc_r+str(alpha))
    # base_mloc_r = base_mloc_r+str(alpha)+"/"
    # base_rloc_r = base_rloc_r+str(alpha)+"/"


    X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
    X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
    #print(y_train_tlsa)
    y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

    #print(y_train_tlsa)
    #return
    for j in range(3):
        obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
        #return
        base_rloc = base_rloc_r + str(j) + "/"
        base_mloc = base_mloc_r + str(j) + "/"
        if not os.path.exists(base_rloc):
            os.mkdir(base_rloc)
        if not os.path.exists(base_mloc):
            os.mkdir(base_mloc)
         
        pdetails = open(base_rloc+"self_kfold_tlsa.txt", 'w')
        tlsa_act = []
        tlsa_pred = []
        for i in range(n_fold):
            #make places to save
            if not os.path.exists(base_rloc+"fold_"+str(i+1)):
                os.mkdir(base_rloc+"fold_"+str(i+1))
            if not os.path.exists(base_mloc+"fold_"+str(i+1)):
                os.mkdir(base_mloc+"fold_"+str(i+1))
            #train the model
            X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
            # print(wrongly_sample_index.shape)
            np.random.seed(j)
            np.random.seed(j)
            X_train_foldi_sub_samp, y_train_foldi_sub_samp, X_train_foldi_ulab, y_train_foldi_ulab = return_labeled_subets(X_train_foldi, y_train_foldi, perc_label)

            tlsa_act.append(y_test_foldi)
            obj_model = progressive_mixup(alpha=alpha)
            if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
                obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

            else:
                obj_model.make_model()
                start_time = datetime.now()
                loss_history = obj_model.train_model(X_train_foldi_sub_samp, y_train_foldi_sub_samp, X_train_foldi_ulab, y_train_foldi_ulab, y_train_foldi_ulab, n_epochs, pdetails, base_mloc)
                end_time = datetime.now()
                print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, X_train_foldi.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
                obj_model.save_model(base_mloc+"fold_"+str(i+1)+"/")
                

                plot_curves(loss_history['loss_conf'], "Conf loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/conf.png", x_label='Epoch')
                plot_curves(loss_history['loss_uncertain'], "Uncertain loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/uncertain.png", x_label='Epoch')
            
            #plot training curves
            pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
            pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
            del obj_model
            
            print(pred_y_train_foldi.shape, pred_y_test_foldi.shape)
            tlsa_pred.append(pred_y_test_foldi)
            
            plot_predictions(pred_y_train_foldi, inv_prep(y_train_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/train_tlsa.png", title_="Train: SFCN finetune", addon= "Age (in years)")
            plot_predictions(pred_y_test_foldi, inv_prep(y_test_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/test_tlsa.png", title_="Test: SFCN finetune", addon= "Age (in years)")
            
        tlsa_act = np.concatenate(tlsa_act)
        tlsa_pred = np.concatenate(tlsa_pred)

        
        plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper_square.png", title_="SFCN: CRAFT", addon="Age")
        pdetails.close()
        # return


if __name__ == "__main__":
    perc_labels = [0.2, 0.4, 0.6]
    for perc_label in perc_labels:
         caller_paper_subsamp_kseed("./Results/rebuttal/mixup_perc_latest/", "./Models/rebuttal/mixup_perc_latest/", perc_label=perc_label, check_train=False)

    # perc_label = 0.2
    # alphas = [0.1, 1.0]
    # for alpha in alphas:
    #    caller_paper_subsamp_kseed("./Results/hyperparameters/datafree_0.2/", "./Models/hyperparameters/datafree_0.2/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", alpha=alpha, check_train=False)
        