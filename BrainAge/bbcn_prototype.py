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
# 	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
# 	try:
# 		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=38000)]) # Notice here
# 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 	except RuntimeError as e:
# 		# Virtual devices must be set before GPUs have been initialized
# 		print(e)


def test_weights(n_bins):
	w = np.arange(1, n_bins+1).astype('float32')
	c = np.arange(n_bins-1)/(n_bins - 2)
	# print(c)
	w_0 = np.zeros(n_bins)
	for i in range(1, n_bins):
		w_0[i] = w_0[i-1] - c[i-1]
	w = tf.constant(np.expand_dims(w, axis=(0,1)))
	w_0 = tf.constant(np.expand_dims(w_0, axis=(0,1)).astype('float32'))
	print(w)
	print(w_0)

class Soft_Binning(tf.keras.layers.Layer):
	#temp is selected from paper
	def __init__(self, n_bins, temp=20, *args, **kwargs):
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
		
		self.w = self.add_weight('w', shape=[1, self.n_bins], initializer=tf.keras.initializers.Constant(np.expand_dims(w, axis=0)), trainable=False)
		self.w_0 = self.add_weight('w', shape=[1, self.n_bins], initializer=tf.keras.initializers.Constant(np.expand_dims(w_0, axis=0).astype('float32')), trainable=False)
		# tf.print(self.w, self.w_0)
	def call(self, inputs):
		#accepts inputs of shape (None, 1)
		# tf.print(inputs)
		out_soft = tf.nn.softmax((self.w*inputs + self.w_0)/self.temp, axis=1)
		# tf.print(inputs, tf.math.reduce_min(out_soft, axis=1), tf.math.reduce_max(out_soft, axis=1), tf.math.argmax(out_soft, axis=1))
		return out_soft
		

class bbcn_sfcn(object):
	def __init__(self, output_shape=1, alpha=0.1, bs=4, lr=0.0001, n_bins = 400):
		self.ymin = 0.0
		self.ymax = 1.0
		self.n_bins = n_bins
		self.alpha = alpha
		self.lr = lr
		self.bs = bs
		self.output_shape = output_shape
		self.model_source = None		#classification based on features - C o F for the source
		self.discrete_net = None
		self.feature_extractor = None
		self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
		self.optimizer_model_t = tf.keras.optimizers.Adam(learning_rate=self.lr)
		self.optimizer_model_common = tf.keras.optimizers.Adam(learning_rate=self.lr)
		self.optimizer_model_common_t = tf.keras.optimizers.Adam(learning_rate=self.lr)
		self.reversal_layer_name = 'reshape'

		self.model_target = None
		self.discrete_net_target = None

		self.prototypes = None
		self.theta = 0.99
	
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
		self.model_source.save(saveloc+"source_model")
		self.model_target.save(saveloc+"target_model")
		
	def load_model(self, saveloc):
		self.model_source = load_model(saveloc+"source_model", compile=False)
		self.model_target = load_model(saveloc+"target_model", compile=False)
		self.model_target.compile()
		self.model_source.compile()
	
	def make_feat_extr_layer(self):
		#this is for the source model
		input_data = Input(shape=self.model_source.inputs[0].shape[1:], name='feat_extr')
		feat_ext_out = self.model_source.get_layer(self.reversal_layer_name).output
		feature_extractor_temp = Model(inputs=self.model_source.inputs, outputs=feat_ext_out, name='feature_extractor_unscaled')
		output_feat_extr = feature_extractor_temp(input_data)
		self.feature_extractor = Model(inputs=input_data, outputs=output_feat_extr, name='feature_extractor')
		self.feature_extractor.compile()
		self.feature_extractor.summary()

		out_discrete = self.model_source(input_data)
		# out_discrete = tf.expand_dims(out_discrete, axis=-1)
		out_discrete = Soft_Binning(name='discretize', n_bins=self.n_bins)(out_discrete)
		# out_discrete = tf.squeeze(out_discrete, axis=1)
		self.discrete_net = Model(inputs=input_data, outputs=out_discrete, name='discrete_net')
		self.discrete_net.compile()
		self.discrete_net.summary()

		# this is for the target model
		input_data = Input(shape=self.model_target.inputs[0].shape[1:], name='target_inp_disc')
		out_discrete = self.model_target(input_data)
		# out_discrete = tf.expand_dims(out_discrete, axis=-1)
		out_discrete = Soft_Binning(name='discretize', n_bins=self.n_bins)(out_discrete)
		# out_discrete = tf.squeeze(out_discrete, axis=1)
		self.discrete_net_target = Model(inputs=input_data, outputs=out_discrete, name='discrete_net_target')
		self.discrete_net_target.compile()
		self.discrete_net_target.summary()

	def make_model(self):
		#get the predictor model
		#the feature extractor output
		mod_object = SFCN_tf(pretrained=True, to_train_full=True)
		self.model_source = mod_object.add_mse_head(True)
		self.model_source.compile()
		self.model_source.summary()
		del mod_object
		mod_object = SFCN_tf(pretrained=True, to_train_full=True)
		self.model_target = mod_object.add_mse_head(True)
		self.model_target.set_weights(self.model_source.get_weights())
		self.model_target.compile()
		self.model_target.summary()
		del mod_object
		
	def predict_mse(self, X):
		all_ops = []
		n_times = int(np.ceil(X.shape[0]/self.bs))
		
		for i in range(n_times):
			if i == n_times - 1:
				end_pos = X.shape[0]
			else:
				end_pos = (i+1)*self.bs
			#print(i*self.hyperparameter['bs_pred'], end_pos)
			
			age_pred = self.model_target(X[i*self.bs : end_pos]).numpy()
			all_ops.append(age_pred.squeeze(axis=-1))
			
		return np.concatenate(all_ops)

	def predict_feats(self, X):
		all_ops = []
		n_times = int(np.ceil(X.shape[0]/self.bs))
		
		for i in range(n_times):
			if i == n_times - 1:
				end_pos = X.shape[0]
			else:
				end_pos = (i+1)*self.bs
			#print(i*self.hyperparameter['bs_pred'], end_pos)
			
			age_pred = self.feature_extractor(X[i*self.bs : end_pos], training=False).numpy()
			all_ops.append(age_pred)
			
		return np.concatenate(all_ops)
	
	def calculate_prototypes(self, X):
		age_ops = []   #stores discrete age
		feat_ops = []
		n_times = int(np.ceil(X.shape[0]/self.bs))
		
		for i in range(n_times):
			if i == n_times - 1:
				end_pos = X.shape[0]
			else:
				end_pos = (i+1)*self.bs
			#print(i*self.hyperparameter['bs_pred'], end_pos)
			
			age_pred = self.discrete_net(X[i*self.bs : end_pos], training=False).numpy()
			feat_pred = self.feature_extractor(X[i*self.bs : end_pos], training=False).numpy()
			age_ops.append(age_pred)
			feat_ops.append(feat_pred)

		feat_ops = np.concatenate(feat_ops)	  # N, d
		age_ops = np.concatenate(age_ops)     # N, N_bins
		# print(age_ops, age_ops.shape)# all_ops.shape)
		
		bin_probs = np.expand_dims(age_ops, axis=-1)	   #N, Nbins, 1
		feats = np.expand_dims(feat_ops, axis=1)	#N, 1, d
		prototypes = np.sum(feats * bin_probs, axis=0, keepdims=False)/(np.sum(bin_probs, axis=0, keepdims=False)+1e-9)	 #N_bins, d / N_bins, 1
		# print(prototypes[0:4])
		return prototypes	  # Nbins, d

	def update_prototypes(self, X):
		if self.prototypes is None:
			self.prototypes = self.calculate_prototypes(X)
			# print(self.prototypes.shape)
		else:
			self.prototypes = self.theta*self.prototypes + (1-self.theta)*self.calculate_prototypes(X)

	def calculate_pseudo_label(self, X):
		features = self.predict_feats(X)  #N,d; protypes is Nbins, d
		
		features = features/(np.sqrt(np.sum(features**2, axis=1, keepdims=True))+1e-9)

		prot_norm = self.prototypes/(np.sqrt(np.sum(self.prototypes**2, axis=1, keepdims=True))+1e-9)

		dot_prod = features @ prot_norm.T   #N,N_bins
		# print(dot_prod)
		# print(dot_prod[0])
		pos_max_sim = np.argmax(dot_prod, axis=1, keepdims=True)  #N, 1
		# print(pos_max_sim, np.max(dot_prod, axis=1, keepdims=True), np.min(dot_prod, axis=1, keepdims=True))
		pseudo_labels = self.ymin + (pos_max_sim + 0.5) * ((self.ymax-self.ymin)/self.n_bins)

		return pseudo_labels 

	def loss_cdist_sup(self, batch_target_lab, batch_y_target_lab, to_train=False):
		
		pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
		loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
		
		return loss_regression_t

	def loss_cdist_usup(self, batch_target_ulab, to_train=False):
		pseudo_labels = self.calculate_pseudo_label(batch_target_ulab)
		
		pred_y_t_ulab = self.model_source(batch_target_ulab, training=to_train)
		# tf.print(pseudo_labels)  #, pred_y_t_ulab)
		loss_regression_t = MeanSquaredError()(pseudo_labels, pred_y_t_ulab)
		
		regularizer = self.alpha*loss_regression_t
		return regularizer

	def loss_cdist_sup_target(self, batch_target_lab, batch_y_target_lab, to_train=False):
		
		pred_y_t_lab = self.model_target(batch_target_lab, training=to_train)
		loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
		
		return loss_regression_t

	def loss_cdist_usup_target(self, batch_target_ulab, to_train=False):
		pred_y_t_ulab = self.discrete_net_target(batch_target_ulab, training=to_train)
		loss_regression_t = tf.math.reduce_mean(-1*pred_y_t_ulab*tf.math.log(pred_y_t_ulab+1e-9))  #N,N_bins
		
		regularizer = self.alpha*loss_regression_t
		return regularizer

	def loss_consistency(self, X, to_train=False):
		source_pred = self.model_source(X, training=to_train)  #N,1
		target_pred = self.model_target(X, training=to_train) #N,1
		loss = tf.math.reduce_mean((source_pred - target_pred)**2)
		return loss

	def back_prop_source(self, batch_target_lab, batch_y_target_lab, batch_target_ulab):
		with tf.GradientTape() as tape:
			#classification loss
			loss_regression_t = self.loss_cdist_sup(batch_target_lab, batch_y_target_lab, to_train=True)
			
		grad_loss_regression = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
		self.optimizer_model_s.apply_gradients(zip(grad_loss_regression , self.model_source.trainable_variables))
		#del grad_class

		with tf.GradientTape() as tape:
			#classification loss
			regularizer = self.loss_cdist_usup(batch_target_ulab, to_train=True)
			
		#update the classifier gradients
		grad_regularizer = tape.gradient(regularizer, self.model_source.trainable_variables)
		self.optimizer_model_s.apply_gradients(zip(grad_regularizer, self.model_source.trainable_variables))
		
		t_loss = loss_regression_t + regularizer
		return t_loss

	def back_prop_target(self, batch_target_lab, batch_y_target_lab, batch_target_ulab):
		with tf.GradientTape() as tape:
			#classification loss
			loss_regression_t = self.loss_cdist_sup_target(batch_target_lab, batch_y_target_lab, to_train=True)
			
		grad_loss_regression = tape.gradient(loss_regression_t, self.model_target.trainable_variables)
		self.optimizer_model_t.apply_gradients(zip(grad_loss_regression , self.model_target.trainable_variables))
		#del grad_class

		with tf.GradientTape() as tape:
			#classification loss
			regularizer = self.loss_cdist_usup_target(batch_target_ulab,  to_train=True)
			
		#update the classifier gradients
		grad_regularizer = tape.gradient(regularizer, self.discrete_net_target.trainable_variables)
		self.optimizer_model_t.apply_gradients(zip(grad_regularizer, self.discrete_net_target.trainable_variables))
		
		t_loss = loss_regression_t + regularizer
		return t_loss

	def back_prop_consistency(self, batch_target_lab, batch_target_ulab):
		with tf.GradientTape(persistent=True) as tape:
			#classification loss
			loss_regression_t = self.loss_consistency(batch_target_lab, to_train=True)
			
		grad_loss_regression = tape.gradient(loss_regression_t, self.model_target.trainable_variables)
		self.optimizer_model_common_t.apply_gradients(zip(grad_loss_regression , self.model_target.trainable_variables))

		grad_loss_regression = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
		self.optimizer_model_common.apply_gradients(zip(grad_loss_regression , self.model_source.trainable_variables))
		del tape

		with tf.GradientTape(persistent=True) as tape:
			#classification loss
			loss_regression_t2 = self.loss_consistency(batch_target_ulab,  to_train=True)
			
		grad_loss_regression = tape.gradient(loss_regression_t2, self.model_target.trainable_variables)
		self.optimizer_model_common_t.apply_gradients(zip(grad_loss_regression , self.model_target.trainable_variables))

		grad_loss_regression = tape.gradient(loss_regression_t2, self.model_source.trainable_variables)
		self.optimizer_model_common.apply_gradients(zip(grad_loss_regression , self.model_source.trainable_variables))
		del tape
		
		t_loss = (loss_regression_t*batch_target_lab.shape[0] + loss_regression_t2*batch_target_ulab.shape[0])/(batch_target_ulab.shape[0] + batch_target_lab.shape[0])
		return t_loss

	def back_prop(self, batch_target_lab, batch_y_target_lab, batch_target_ulab):
		self.update_prototypes(np.concatenate([batch_target_ulab, batch_target_ulab]))
		l_sma = self.back_prop_source(batch_target_lab, batch_y_target_lab, batch_target_ulab)
		l_tml = self.back_prop_target(batch_target_lab, batch_y_target_lab, batch_target_ulab)
		l_cons = self.back_prop_consistency(batch_target_lab, batch_target_ulab)
		return l_sma, l_tml, l_cons

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
		# y = np.squeeze(y)
		indices = np.arange(y.shape[0])
		# indices = np.argsort(y)
		np.random.shuffle(indices)
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
	# def train_model(self, X_target, y_target, y_gmm_data, num_epochs, printDetails):
	def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_train_y, num_epochs, printDetails, m_saveloc):
		# print(discrete_distr.shape)
		# sys.exit()
		loss_history = {
						'loss_sma': [],
						'loss_tml': [],
						'loss_cons': [],
					   }
		# train_data_target_t_lab = np.arange(X_target.shape[0])
		# train_data_target_t, last_batch_sz = self.split_2d(y_target, self.bs)
		train_data_target_t, last_batch_sz = self.split_2d(y_target_ulab, self.bs)

		# ep_length = int(np.ceil(len(X_target)/self.bs))
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
				if end_at_target_lab < start_from_target_lab:
					domain_selector_lab = np.concatenate([train_data_target_t_lab[start_from_target_lab:len(train_data_target_t_lab)], train_data_target_t_lab[0:end_at_target_lab]])
					target_ctr = 0
				else:
					domain_selector_lab = train_data_target_t_lab[start_from_target_lab:end_at_target_lab]
					target_ctr += 1	 

				batch_X_target_lab = X_target_lab[domain_selector_lab]
				batch_y_target_lab = y_target_lab[domain_selector_lab]

				loss_sma, loss_tml, loss_cons = self.back_prop(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab)
				#keep track of the loss
				loss_history['loss_sma'].append(loss_sma.numpy())
				loss_history['loss_tml'].append(loss_tml.numpy())
				loss_history['loss_cons'].append(loss_cons.numpy())
				#print(loss_history['loss_regression_t'], loss_history['loss_t'])   
			if i%2 == 0:
				#import pdb;pdb.set_trace()
				print("Losses at epoch {}, Loss SMA: {}, Loss TML: {}, Loss cons: {}".format(i+1, loss_history['loss_sma'][-2], loss_history['loss_tml'][-2], loss_history['loss_cons'][-2]), file=printDetails, flush=True)
				print("Losses at epoch {}, Loss SMA: {}, Loss TML: {}, Loss cons: {}".format(i+1, loss_history['loss_sma'][-2], loss_history['loss_tml'][-2], loss_history['loss_cons'][-2]))
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
	n_epochs = 15	  #keep the no of updates fixed
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"
	if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
		 os.mkdir(base_rloc_r+str(int(perc_label*100)))
	if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
		 os.mkdir(base_mloc_r+str(int(perc_label*100)))
	base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
	base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"
	# if not os.path.exists(base_rloc_r+str(alpha)):
	# 	os.mkdir(base_rloc_r+str(alpha))
	# if not os.path.exists(base_mloc_r+str(alpha)):
	# 	os.mkdir(base_mloc_r+str(alpha))
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
			X_train_foldi_sub_samp, y_train_foldi_sub_samp, _, _ = return_labeled_subets(X_train_foldi, y_train_foldi, perc_label)

			tlsa_act.append(y_test_foldi)
			obj_model = bbcn_sfcn(alpha=alpha)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				obj_model.make_model()
				obj_model.make_feat_extr_layer()
				start_time = datetime.now()
				loss_history = obj_model.train_model(X_train_foldi_sub_samp, y_train_foldi_sub_samp, X_train_foldi, y_train_foldi, y_train_foldi, n_epochs, pdetails, base_mloc)
				end_time = datetime.now()
				print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, X_train_foldi.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
				obj_model.save_model(base_mloc+"fold_"+str(i+1)+"/")
				

				plot_curves(loss_history['loss_sma'], "SMA loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/sma.png", x_label='Epoch')
				plot_curves(loss_history['loss_tml'], "TML loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/tml.png", x_label='Epoch')
				plot_curves(loss_history['loss_cons'], "Cons loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/cons.png", x_label='Epoch')
			
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
		 caller_paper_subsamp_kseed("./Results/rebuttal/bbcn_perc/", "./Models/rebuttal/bbcn_perc/", perc_label=perc_label, check_train=False)

	# perc_label = 0.2
	# alphas = [0.1, 1.0]
	# for alpha in alphas:
	#	 caller_paper_subsamp_kseed("./Results/hyperparameters/datafree_0.2/", "./Models/hyperparameters/datafree_0.2/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", alpha=alpha, check_train=False)
		