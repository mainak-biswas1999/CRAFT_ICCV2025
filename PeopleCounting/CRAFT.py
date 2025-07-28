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


class CRAFT(object):
    
    def __init__(self, output_shape=1, alpha=0.1, bs=5, lr=0.0001, n_bins=500, p_y_gmm=None):
                                    
        self.p_y_gmm = p_y_gmm
        self.alpha = alpha
        self.lr = lr
        self.bs = bs
        self.output_shape = output_shape
        self.model_source = None        #classification based on features - C o F for the source
        self.N_total = None
        
        self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        self.range_target = None
        self.target_labels = None
        self.exp_diff_prior = None
        self.n_bins = n_bins
        
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
        self.model_source.save(saveloc+"attn-resnet")
        
    def load_model(self, saveloc):
        self.model_source = load_model(saveloc+"attn-resnet", compile=False)
        # self.model_source = load_model(saveloc, compile=False)
        self.model_source.compile()
        self.model_source.summary()
    
    #works only at inference
    def predict(self, X):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.bs))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.bs
            
            all_ops.append(self.model_source(X[i*self.bs : end_pos]).numpy())
        
        return np.concatenate(all_ops)

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
        
    def get_pseudo_labels_bins(self, target_X):
        target_pred = self.model_source(target_X, training=False)
        exp_diff_posterior = tf.cast(tf.math.exp(-(target_pred - self.age_labels)**2), 'float32')
        
        cdist_rescaled_posteriors = (exp_diff_posterior * self.exp_diff_prior)/(tf.math.reduce_sum(exp_diff_posterior, axis=0, keepdims=True))
        pos_max = tf.cast(tf.math.argmax(cdist_rescaled_posteriors, axis=1), 'float32')
        psuedo_labels = tf.expand_dims(self.range_target[0] + (pos_max + 0.5) * (self.range_target[1] - self.range_target[0]) / self.n_bins, axis=1)
        
        return psuedo_labels

    def loss_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=False):
        pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
        pred_y_t_ulab = self.model_source(batch_target_ulab, training=to_train)
        psuedo_labels = self.get_pseudo_labels_bins(batch_target_ulab)
        #uncomment this
        loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
        loss_t_on_source = MeanSquaredError()(psuedo_labels, pred_y_t_ulab)
        pred_y_t_regu = tf.repeat(tf.transpose(pred_y_t_ulab), repeats=[batch_y_target_ulab.shape[0]], axis=0)
        #mse
        # regu_diff = tf.math.exp(-(pred_y_t_regu - psuedo_labels)**2)
        regu_diff = -(pred_y_t_regu - psuedo_labels)**2
        # sum_ydiff_for_eachX = tf.math.log((self.N_total/batch_target.shape[0])*tf.math.reduce_sum(regu_diff, axis=1))
        #constants don't change the optimization
        #log-exp trick

        max_terms = tf.math.reduce_max(regu_diff, axis=1, keepdims=True)
        exponentiate = tf.math.exp(regu_diff - max_terms)
        sum_ydiff_for_eachX = max_terms + tf.math.log(tf.math.reduce_sum(exponentiate, axis=1, keepdims=True))  
        regularizor = (loss_t_on_source + tf.math.reduce_mean(sum_ydiff_for_eachX))

        t_loss = loss_regression_t + self.alpha * regularizor
        
        return t_loss, loss_regression_t
    
    def back_prop_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab):
        
        with tf.GradientTape(persistent=True) as tape:
            #classification loss
            t_loss, loss_regression_t = self.loss_cdist(batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=True)
            
        #update the classifier gradients
        grad_class = tape.gradient(t_loss, self.model_source.trainable_variables)
        
        self.optimizer_model_s.apply_gradients(zip(grad_class, self.model_source.trainable_variables))
        return t_loss, loss_regression_t

    def make_perms(self, stratify_2D, last_batch_sz):
        for i in range(stratify_2D.shape[1]):
            if np.random.rand() > 0.5:
                if i>=last_batch_sz:
                    permutation = np.concatenate([np.random.permutation(stratify_2D.shape[0]-1), [stratify_2D.shape[0]-1]])
                    # print(permutation)
                else:
                    permutation = np.random.permutation(stratify_2D.shape[0])
            else:
                permutation = np.arange(stratify_2D.shape[0])
            # print(permutation)

            stratify_2D[:, i] = stratify_2D[permutation, i]
        
        # print(last_batch_sz, stratify_2D.astype('int'))
        
        #print(stratify_2D.astype('int'))
        return stratify_2D.astype('int') 


    def split_2d(self, y, bs):
        y = np.squeeze(y)
        indices = np.argsort(y)
        # print(indices)

        n_cols = bs
        n_rows = int(np.ceil(len(y)/bs))
        
        last_batch_sz = n_cols - (n_rows*n_cols - len(y))
        stratify_2D = np.zeros((n_rows, n_cols))

        ctr = 0
        for i in range(stratify_2D.shape[1]):
            for j in range(stratify_2D.shape[0]):
                if i>=last_batch_sz and j==stratify_2D.shape[0]-1:
                    continue
                stratify_2D[j, i] = indices[ctr]
                # print(i, j, ctr)
                ctr += 1
        # print(ctr)
        # print(y[stratify_2D.astype('int')], last_batch_sz, stratify_2D.shape)
        return stratify_2D, last_batch_sz

    def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_val_X, target_val_y, num_epochs, printDetails, m_saveloc):
        
        self.range_target = [0.0, 1.0]
        create_labels = (np.expand_dims(np.linspace(self.range_target[0], self.range_target[1], self.n_bins), axis=0) + 0.5*(self.range_target[1] - self.range_target[0])/(self.n_bins - 1)).astype('float32')
        # print(create_labels.shape)
        
        if self.p_y_gmm is None:    
            p_y_gmm = EM_Exp_GMM_1d(2, 2, y_target_lab)    # from 0 onwards to make the exponential work
            p_y_gmm.run_EM()
        
        densities_py = self.p_y_gmm.return_P(create_labels.T).T
        self.plot_histo(create_labels.ravel(), densities_py.ravel(), "r (in pix) - scaled", "Density", "del r", "./results/stats/gmm_0_1_wm.png")
        # sys.exit()
        # print(densities_py.shape)
        self.age_labels = tf.constant(create_labels)
        self.exp_diff_prior = tf.cast(densities_py, 'float32')
        # print(self.exp_diff_prior.shape, self.age_labels.shape)
        # sys.exit()

        loss_history = {
                        'loss_regression_t_total': [],
                        'loss_t': [],
                        'val_rmse': []
                         }
        

        #train_data_target_t = np.arange(X_target.shape[0])
        
        train_data_target_t, last_batch_sz = self.split_2d(y_target_ulab, self.bs)

        #ep_length = int(np.ceil(len(X_target)/self.bs))
        target_ctr = 0
        best_rmse = 1e11
        train_data_target_t_lab = np.arange(X_target_lab.shape[0])
        best_rmse_pos = 0
        for i in range(num_epochs):
            train_data_target_t = self.make_perms(train_data_target_t, last_batch_sz)
            np.random.shuffle(train_data_target_t_lab)
            for j in range(train_data_target_t.shape[0]):
                
                if j==train_data_target_t.shape[0]-1:
                    domain_selector = train_data_target_t[j, 0:last_batch_sz]
                else:
                    domain_selector = train_data_target_t[j]

                # print(train_data_target_t.shape, len(domain_selector))
                batch_X_target_ulab = X_target_ulab[domain_selector]
                #important common mistake (N, ) and (N, 1) mismatch
                batch_y_target_ulab = y_target_ulab[domain_selector]

                start_from_target_lab = (target_ctr*self.bs) % len(y_target_lab)
                end_at_target_lab = ((target_ctr+1)*self.bs) % len(y_target_lab)
                
                if len(train_data_target_t_lab) < self.bs:
                    domain_selector_lab = train_data_target_t_lab

                elif end_at_target_lab < start_from_target_lab:
                    domain_selector_lab = train_data_target_t_lab[start_from_target_lab:len(train_data_target_t_lab)] #np.concatenate([train_data_target_t_lab[start_from_target_lab:len(train_data_target_t_lab)], train_data_target_t_lab[0:end_at_target_lab]])
                    np.random.shuffle(train_data_target_t_lab)
                    target_ctr = 0
                else:
                    domain_selector_lab = train_data_target_t_lab[start_from_target_lab:end_at_target_lab]
                    target_ctr += 1     

                batch_X_target_lab = X_target_lab[domain_selector_lab]
                batch_y_target_lab = y_target_lab[domain_selector_lab]
                # print(train_data_target_t_lab.shape, batch_X_target_lab.shape, train_data_target_t.shape, batch_X_target_ulab.shape)
                # sys.exit()
                t_loss, loss_regression_t = self.back_prop_cdist(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab, batch_y_target_ulab)
                #keep track of the loss
                loss_history['loss_regression_t_total'].append(loss_regression_t.numpy())
                loss_history['loss_t'].append(t_loss.numpy())


            pred_val_y = self.predict(target_val_X)
            loss_history['val_rmse'].append(get_rmse1(target_val_y, pred_val_y))
            # if loss_history['val_rmse'][-1] < best_rmse:
            #   best_rmse = loss_history['val_rmse'][-1]
            #   self.save_model(m_saveloc)
            if (loss_history['val_rmse'][-1] < best_rmse):  #or ((loss_history['val_rmse'][-1] < 1.05*best_rmse) and (i - best_rmse_pos > 0.5*num_epochs)):
                best_rmse_pos = i
                best_rmse = loss_history['val_rmse'][-1]
                self.save_model(m_saveloc)  
            
            if i%2 == 0:
                #import pdb;pdb.set_trace()
                print("Losses at epoch {}, Total T loss: {}, Target Reg. Loss: {}, val rmse: {}".format(i+1, loss_history['loss_t'][-2], loss_history['loss_regression_t_total'][-2], loss_history['val_rmse'][-1]), file=printDetails, flush=True)
                print("Losses at epoch {}, Total T loss: {}, Target Reg. Loss: {}, Val rmse: {}".format(i+1, loss_history['loss_t'][-2], loss_history['loss_regression_t_total'][-2], loss_history['val_rmse'][-1]))

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
    n_epochs = 10
    pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
    np.random.seed(seed)

    target_train_X_lab, target_train_y_lab, _, _ = return_labeled_subets(target_train_X, target_train_y, perc_label) 
    # target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)

    obj_cdist = CRAFT(p_y_gmm=p_y_gmm)
    obj_cdist.make_model(pretrain_loc)
    
    start_time = datetime.now()
    loss_history = obj_cdist.train_model(target_train_X_lab, target_train_y_lab, target_train_X, target_train_y, target_val_X, target_val_y, n_epochs, pdetails, m_saveloc)
    end_time = datetime.now()
    print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
    # obj_cdist.save_model(m_saveloc)
    plot_curves(loss_history['loss_regression_t_total'], "Total Target Loss (incl. pseudoloss)", "loss", n_epochs, r_saveloc+"/regression_target.png", x_label='Epoch')
    plot_curves(loss_history['loss_t'], "Target Regression Loss", "loss", n_epochs, r_saveloc+"/total_target_loss.png", x_label='Epoch')
    plot_curves(loss_history['val_rmse'], "Validation Error", "Error", n_epochs, r_saveloc+"/val_rmse.png", x_label='Epoch')
    
    pdetails.close()
    del target_train_X, target_train_y, target_val_X, target_val_y
    del obj_cdist

def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, m_saveloc, r_saveloc):
    obj_cdist = CRAFT()
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
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Attn-Resnet: CRAFT', addon=" # People")

    del obj_cdist


if __name__=='__main__':
    import gc
    
    base_mloc = "./models/CRAFT/"
    base_rloc = "./results/CRAFT/"
    source_loc = "./models/semi_sup/"
    _, _, _, _, _, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people(log_scale=False)
    p_y_gmm = EM_Exp_GMM_1d(2, 2, y_train_ood)
    p_y_gmm.run_EM()

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
            run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", p_y_gmm, pretrain_loc="./models/nwpu_base_model/", seed=j, perc_label=(0.01*i))
            run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            
            gc.collect()
            tf.keras.backend.clear_session()