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



class Soft_Binning(tf.keras.layers.Layer):
    #temp is selected from paper
    def __init__(self, n_bins, temp=0.01, *args, **kwargs):
        #input size should have the number of features in each modality - works for 1d feature vectors only.
        super(Soft_Binning, self).__init__(*args, **kwargs)
        self.temp = temp
        self.n_bins = n_bins
    
    def build(self, input_shape):
        w = np.arange(1, self.n_bins+1).astype('float32')
        c = np.arange(self.n_bins-1)/(self.n_bins - 2)
        # print(c)
        w_0 = np.zeros(self.n_bins)
        for i in range(1, self.n_bins):
            w_0[i] = w_0[i-1] - c[i-1]
        
        self.w = self.add_weight('w', shape=[1, 1, self.n_bins], initializer=tf.keras.initializers.Constant(np.expand_dims(w, axis=(0,1))), trainable=False)
        self.w_0 = self.add_weight('w', shape=[1, 1, self.n_bins], initializer=tf.keras.initializers.Constant(np.expand_dims(w_0, axis=(0,1)).astype('float32')), trainable=False)
        
    def call(self, inputs):
        #accepts inputs of shape (None, f, 1)
        out_soft = tf.nn.softmax((self.w*inputs + self.w_0)/self.temp)
        return out_soft

class DataFree(object):
    def __init__(self, output_shape=1, alpha=0.1, bs=48, lr=0.0001, n_bins=400):
                                     # L0,  L1, L2, L3, op
        
        self.alpha = alpha
        self.lr = lr
        self.bs = bs
        self.output_shape = output_shape
        self.model_source = None        #classification based on features - C o F for the source
        self.feature_extractor = None
        self.N_total = None
        
        self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
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

    def make_feat_extr_layer(self):
        
        input_data = Input(shape=self.model_source.inputs[0].shape[1:], name='feat_extr')
        feat_ext_out = self.model_source.get_layer('Dense_l1').output
        feature_extractor_temp = Model(inputs=self.model_source.inputs, outputs=feat_ext_out, name='feature_extractor_unscaled')
        output_feat_extr = feature_extractor_temp(input_data)
        output_feat_extr = tf.keras.activations.sigmoid(output_feat_extr)
        output_feat_extr = tf.expand_dims(output_feat_extr, axis=-1)
        output_feat_extr = Soft_Binning(name='discretize', n_bins=self.n_bins)(output_feat_extr)

        self.feature_extractor = Model(inputs=input_data, outputs=output_feat_extr, name='feature_extractor')
        self.feature_extractor.compile()
        self.feature_extractor.summary()

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
    
    def predict_feats(self, X):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.bs))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.bs
            #print(i*self.hyperparameter['bs_pred'], end_pos)
            
            age_pred = self.feature_extractor(X[i*self.bs : end_pos]).numpy()
            all_ops.append(age_pred)
            
        return np.concatenate(all_ops)

    def get_source_distr(self, X_target_lab):
        latents = self.predict_feats(X_target_lab)
        # print(latents.shape)
        return (np.sum(latents, axis=0)/X_target_lab.shape[0]  + 1e-9)

    def loss_cdist_sup(self, batch_target_lab, batch_y_target_lab, to_train=False):
        
        pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
        loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
        
        return loss_regression_t

    def loss_cdist_usup(self, batch_target_ulab, source_distr, to_train=False):
        pred_ulab = self.feature_extractor(batch_target_ulab, training=to_train)
        target_distr = tf.math.reduce_mean(pred_ulab, axis=0) + 1e-9
        # tf.print(target_distr.shape)
        JS_div = tf.math.reduce_mean(0.5*(source_distr*tf.math.log(source_distr/target_distr) + target_distr*tf.math.log(target_distr/source_distr)), axis=-1)
        # tf.print(JS_div.shape)
        # sys.exit()
        regularizer = self.alpha*tf.math.reduce_mean(JS_div)
        return regularizer


    def back_prop_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, discrete_distr):
        
        with tf.GradientTape() as tape:
            #classification loss
            loss_regression_t = self.loss_cdist_sup(batch_target_lab, batch_y_target_lab, to_train=True)
            
        grad_loss_regression = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
        self.optimizer_model_s.apply_gradients(zip(grad_loss_regression , self.model_source.trainable_variables))
        #del grad_class

        with tf.GradientTape() as tape:
            #classification loss
            regularizer = self.loss_cdist_usup(batch_target_ulab, discrete_distr, to_train=True)
            
        #update the classifier gradients
        grad_regularizer = tape.gradient(regularizer, self.feature_extractor.trainable_variables)
        self.optimizer_model_s.apply_gradients(zip(grad_regularizer, self.feature_extractor.trainable_variables))
        
        t_loss = loss_regression_t + regularizer
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
        discrete_distr = self.get_source_distr(X_target_lab)
        
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
                t_loss, loss_regression_t = self.back_prop_cdist(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab, discrete_distr)
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


def run_finetune(target_train_X, target_train_y, target_val_X, target_val_y, m_saveloc, r_saveloc, pretrain_loc, seed=0, perc_label=0.5, alpha=0.1):
    n_epochs = 10
    pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
    np.random.seed(seed)

    target_train_X_lab, target_train_y_lab, target_train_X_ulab, target_train_y_ulab = return_labeled_subets(target_train_X, target_train_y, perc_label) 
    # target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)

    obj_cdist = DataFree(alpha=alpha)
    obj_cdist.make_model(pretrain_loc)
    obj_cdist.make_feat_extr_layer()
    
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

def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, m_saveloc, r_saveloc):
    obj_cdist = DataFree()
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
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Resnet: DataFree', addon=" # People")

    del obj_cdist


if __name__=='__main__':
    import gc
    
    base_mloc = "./models/DataFree/"
    base_rloc = "./results/DataFree/"
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
            run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", pretrain_loc=source_loc+str(i)+"/"+str(j)+"/", seed=j, perc_label=(0.01*i))
            run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            
            gc.collect()
            tf.keras.backend.clear_session()