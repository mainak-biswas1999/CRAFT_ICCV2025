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
# from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
from datetime import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')


if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40000)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)    

matplotlib.use('Agg') 

class EM_Exp_GMM_1d:
    def __init__(self, n_g, n_e, X):
        self.n_g = n_g  #number of gaussians = k1
        self.n_e = n_e  #number of exponentials = k2
        
        self.X = X      #(N, 1)
        # the variational distribution q(z = j | X_i) = q_ji -> dimension (N, k1) - for normal, (N, k2) - for normal
        self.q1 = None
        self.q2 = None
        #dimension (k1, 1) -> k normals, univariate
        self.mus = None
        #dimension (k1, 1) -> k normals, univariate 
        self.sigmas = None
        #lambdas (k2, 1) -> for exponential distribution
        self.lambdas = None
        #multinomial - (k1+k2,)
        self.alphas = None #k1 gaussians
        self.betas = None   #k2 exponentials
        
        #initialize the parameters
        self.initialize_params()
        print("Data dimension: {} \nDim. alphas: {} \nDim. mus: {}, \nDim. sigmas: {} \nDim. lambdas: {} \nDim q1: {} \nDim q2: {}".format(self.X.shape, self.alphas.shape, self.mus.shape, self.sigmas.shape, self.lambdas.shape, self.q1.shape, self.q2.shape))
    
    def initialize_params(self):
        np.random.seed(99)
        #list you want to append
        mus = []
        # for the features randomly set the means of the distributions
        #covariance of the data
        sigma_data = np.std(self.X.ravel())
        
        for i in range(self.X.shape[1]):
            mus.append(np.random.normal(loc=np.mean(self.X[:, i]), scale=np.std(self.X[:, i]), size=(self.n_g, 1)))
            
        #The sigmas of all the normals is set to the covariance of the data 
        self.sigmas = np.repeat(np.expand_dims(sigma_data, axis=(0, 1)), self.n_g, axis=0)
        self.alphas = np.repeat(1./self.n_g, self.n_g)
        #setting uniform probabilities for each datapoint in the variational distribution
        self.q1 = np.repeat(np.expand_dims(self.alphas, axis=0), self.X.shape[0], axis=0)
        self.mus = np.concatenate(mus, axis=1)
        
        self.betas = np.repeat(1./self.n_e, self.n_e)
        #setting uniform probabilities for each datapoint in the variational distribution
        self.q2 = np.repeat(np.expand_dims(self.betas, axis=0), self.X.shape[0], axis=0)
        self.lambdas = np.ones((self.n_e, 1))

    def get_all_normal_values(self, X):
        # get the normal pdf value of all the data points for all the normals in the GMM (N, k1) 
        all_normal_vals = np.zeros((X.shape[0], self.n_g))
        
        for i in range(self.n_g):
            #calculate the normal values for all the data points and a single normal
            for j in range(X.shape[0]):
                exp_term_ji = np.exp(-0.5* ((X[j, 0] - self.mus[i, 0])/self.sigmas[i, 0])**2)
                all_normal_vals[j, i] = (1./(np.sqrt(2*np.pi) * self.sigmas[i, 0])) * exp_term_ji
            
        return all_normal_vals
    
    def get_all_exp_values(self, X):
        # get the exponential pdf value of all the data points for all Exp. (N, k2) 
        all_exp_vals = np.zeros((X.shape[0], self.n_e))
        
        for i in range(self.n_e):
            #calculate the normal values for all the data points and a single normal
            for j in range(X.shape[0]):
                all_exp_vals[j, i] = self.lambdas[i, 0] * np.exp(-self.lambdas[i, 0] * X[j, 0]) * int(X[j, 0] > 0.)
        
        return all_exp_vals
    
    def Expectation(self):
        all_normal_vals = self.get_all_normal_values(self.X)
        all_exp_vals = self.get_all_exp_values(self.X)
        #calculate the variational distribution: dimension of q: (N, k)
        #print(np.expand_dims(self.alphas, axis=0).shape)
        q1_alpha_prod = np.expand_dims(self.alphas, axis=0) * all_normal_vals 
        q2_alpha_prod = np.expand_dims(self.betas, axis=0) * all_exp_vals
        #normalize all the qs by the sum across k normals
        norm_val = np.sum(q1_alpha_prod, axis=1, keepdims=True) + np.sum(q2_alpha_prod, axis=1, keepdims=True)
        self.q1 = q1_alpha_prod / norm_val
        self.q2 = q2_alpha_prod / norm_val

    def Maximization(self):
        #set the variational distribution q
        self.alphas = np.mean(self.q1, axis=0)
        self.betas = np.mean(self.q2, axis=0)
        #print(np.sum(self.alphas))
        norm_term = np.expand_dims(self.X.shape[0] * self.alphas, axis=1)   #converting it to k1x1
        self.mus = (self.q1.T @ self.X) / norm_term
        #sigmas = np.zeros(self.sigmas.shape)
        for i in range(self.n_g):
            #X has examples in the rows
            mu_i = np.expand_dims(self.mus[i, 0], axis=0)
            #\|q (X-u)
            X_tilde = (self.X - mu_i) * np.sqrt(np.expand_dims(self.q1[:, i], axis=1))
            self.sigmas[i, 0] = np.sqrt((X_tilde.T @ X_tilde)[0, 0] / norm_term[i, 0]) 
        #exponentials - 
        norm_term2 = np.expand_dims(self.X.shape[0] * self.betas, axis=1)   #converting it to k2x1
        self.lambdas = norm_term2 / (self.q2.T @ self.X) 
         

    def run_EM(self, max_iter=100):
        for i in range(max_iter):
            self.Expectation()
            self.Maximization()
    
    def return_P(self, X):
        P_normal = self.get_all_normal_values(X)
        P_exp = self.get_all_exp_values(X)
        P = P_exp @ np.expand_dims(self.betas, axis=1) +    P_normal @ np.expand_dims(self.alphas, axis=1)
        return P



class progressive_mixup(object):

    def __init__(self, type_='r', output_shape=1, alpha=0.01, bs=36, lr=0.0001, n_bins=400):
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
    
    def predict(self, X, bs=36):
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


def run_finetune(target_train_X, target_train_y, target_val_X, target_val_y, m_saveloc, r_saveloc, p_y_gmm, pretrain_loc, seed=0, perc_label=0.5, alpha=0.1):
    n_epochs = 5
    pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
    np.random.seed(seed)

    target_train_X_lab, target_train_y_lab, _, _ = return_labeled_subets(target_train_X, target_train_y, perc_label) 
    # target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)

    obj_cdist = progressive_mixup(alpha=alpha)
    obj_cdist.make_model(pretrain_loc)
    
    start_time = datetime.now()
    loss_history = obj_cdist.train_model(target_train_X_lab, target_train_y_lab, target_train_X, target_train_y, target_val_X, target_val_y, n_epochs, pdetails, m_saveloc)
    end_time = datetime.now()
    print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
    # obj_cdist.save_model(m_saveloc)
    plot_curves(loss_history['loss_conf'], "Confident Loss", "loss", n_epochs, r_saveloc+"/conf_loss.png", x_label='Epoch')
    plot_curves(loss_history['loss_uncertain'], "Uncertain Loss", "loss", n_epochs, r_saveloc+"/uncertain_loss.png", x_label='Epoch')
    plot_curves(loss_history['val_rmse'], "Validation Error", "Error", n_epochs, r_saveloc+"/val_rmse.png", x_label='Epoch')
    
    pdetails.close()
    del target_train_X, target_train_y, target_val_X, target_val_y
    del obj_cdist

def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, m_saveloc, r_saveloc):
    obj_cdist = progressive_mixup()
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
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Attn-Resnet: CRAFT', addon=" # People")

    del obj_cdist


if __name__=='__main__':
    import gc
    
    base_mloc = "./models/Mixup/"
    base_rloc = "./results/Mixup/"
    source_loc = "./models/semi_sup/"

    X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood = read_cam('/data1/mainak/cancer_pred/data/cam17.npz')
    p_y_gmm = GaussianMixture(n_components=4, random_state=0).fit(y_train_ood)

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
            run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", p_y_gmm, pretrain_loc=source_loc+str(i)+"/"+str(j)+"/", seed=j, perc_label=(0.01*i))
            run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            
            gc.collect()
            tf.keras.backend.clear_session()