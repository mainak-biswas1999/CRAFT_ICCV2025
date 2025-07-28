from loader import *
from LSTM_EEGNet import *
from pyramidal_cnn import *
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
from plot_paper import *
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
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)]) # Notice here
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
		#for -1 to 1 -- in lstm
		out_soft = tf.nn.softmax((0.5*self.w*(inputs+1.0) + self.w_0)/self.temp, axis=1)
		return out_soft
		
class bbcn_eye(object):
	def __init__(self, type_='r', model_name='EEGNET-LSTM', output_shape=1, alpha=0.1, bs=128, lr=0.0001, n_bins = 400, target_dataset='DDir'):
		self.ymin = -1.0
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

		self.target_dataset = target_dataset
		self.model_name = model_name
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
		self.model_source.save(saveloc+"source_model")
		self.model_target.save(saveloc+"target_model")
		
	def load_model(self, saveloc):
		self.model_source = load_model(saveloc+"source_model", compile=False)
		self.model_target = load_model(saveloc+"target_model", compile=False)
		self.model_target.compile()
		self.model_source.compile()
	
	def make_feat_extr_layer(self):
		if self.model_name == 'EEGNET-LSTM':
			input_data = Input(shape=self.model_source.inputs[0].shape[1:], name='feat_extr')
			feat_ext_out = self.model_source.get_layer('lstm_batchnorm').output
			feature_extractor_temp = Model(inputs=self.model_source.inputs, outputs=feat_ext_out, name='feature_extractor_unscaled')
			output_feat_extr = feature_extractor_temp(input_data)
			output_feat_extr = Flatten()(output_feat_extr)
			self.feature_extractor = Model(inputs=input_data, outputs=output_feat_extr, name='feature_extractor')
			self.feature_extractor.compile()
			self.feature_extractor.summary()
		else:
			input_data = Input(shape=self.model_source.inputs[0].shape[1:], name='feat_extr')
			feat_ext_out = self.model_source.get_layer('global_average_pooling1d').output
			feature_extractor_temp = Model(inputs=self.model_source.inputs, outputs=feat_ext_out, name='feature_extractor_unscaled')
			output_feat_extr = feature_extractor_temp(input_data)
			self.feature_extractor = Model(inputs=input_data, outputs=output_feat_extr, name='feature_extractor')
			self.feature_extractor.compile()
			self.feature_extractor.summary()

		input_data = Input(shape=self.model_target.inputs[0].shape[1:], name='target_inp_disc')
		out_discrete = self.model_source(input_data)
		# out_discrete = tf.expand_dims(out_discrete, axis=-1)
		out_discrete = Soft_Binning(name='discretize', n_bins=self.n_bins)(out_discrete)
		# out_discrete = tf.squeeze(out_discrete, axis=1)
		self.discrete_net = Model(inputs=input_data, outputs=out_discrete, name='discrete_net')
		self.discrete_net.compile()
		self.discrete_net.summary()

		# this is for the target model
		input_data = Input(shape=self.model_target.inputs[0].shape[1:], name='target_inp_disc_target')
		out_discrete = self.model_target(input_data)
		# out_discrete = tf.expand_dims(out_discrete, axis=-1)
		out_discrete = Soft_Binning(name='discretize', n_bins=self.n_bins)(out_discrete)
		# out_discrete = tf.squeeze(out_discrete, axis=1)
		self.discrete_net_target = Model(inputs=input_data, outputs=out_discrete, name='discrete_net_target')
		self.discrete_net_target.compile()
		self.discrete_net_target.summary()
	
	def make_model(self, pretrain_loc=None):
		#get the predictor model
		#the feature extractor output
		if pretrain_loc is None:
			if self.model_name == 'EEGNET-LSTM':
				enet_lstm_obj = LSTM_EEGNet(type_=self.type_)
			else:
				enet_lstm_obj = SpyrCNN_wrapper(type_=self.type_)
			self.model_source = enet_lstm_obj.make_model(True)
			del enet_lstm_obj
		else:
			if self.model_name == 'EEGNET-LSTM':
				enet_lstm_obj = LSTM_EEGNet(type_=self.type_)
			else:
				enet_lstm_obj = SpyrCNN_wrapper(type_=self.type_)
			self.model_source = enet_lstm_obj.load_model(pretrain_loc, True)
			del enet_lstm_obj
		self.model_source.compile()
		self.model_source.summary()

		if pretrain_loc is None:
			if self.model_name == 'EEGNET-LSTM':
				enet_lstm_obj = LSTM_EEGNet(type_=self.type_)
			else:
				enet_lstm_obj = SpyrCNN_wrapper(type_=self.type_)
			self.model_target = enet_lstm_obj.make_model(True)
			del enet_lstm_obj
		else:
			if self.model_name == 'EEGNET-LSTM':
				enet_lstm_obj = LSTM_EEGNet(type_=self.type_)
			else:
				enet_lstm_obj = SpyrCNN_wrapper(type_=self.type_)
			self.model_target = enet_lstm_obj.load_model(pretrain_loc, True)
			del enet_lstm_obj
		self.model_target.compile()
		self.model_target.summary()
		
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
		age_ops = np.concatenate(age_ops)	 # N, N_bins
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
	def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_val_X, target_val_y, target_train_y, num_epochs, printDetails, m_saveloc):
		# print(discrete_distr.shape)
		# sys.exit()
		loss_history = {
						'loss_sma': [],
						'loss_tml': [],
						'loss_cons': [],
						'val_rmse': [],
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
			pred_val_y = self.model_target.predict(target_val_X)
			loss_history['val_rmse'].append(get_rmse1(target_val_y, pred_val_y))
			if (loss_history['val_rmse'][-1] < best_rmse):  #or ((loss_history['val_rmse'][-1] < 1.05*best_rmse) and (i - best_rmse_pos > 0.5*num_epochs)):
				best_rmse_pos = i
				best_rmse = loss_history['val_rmse'][-1]
				self.save_model(m_saveloc)
			if i%2 == 0:
				#import pdb;pdb.set_trace()
				print("Losses at epoch {}, Loss SMA: {}, Loss TML: {}, Loss cons: {},  val rmse: {}".format(i+1, loss_history['loss_sma'][-2], loss_history['loss_tml'][-2], loss_history['loss_cons'][-2], loss_history['val_rmse'][-1]), file=printDetails, flush=True)
				print("Losses at epoch {}, Loss SMA: {}, Loss TML: {}, Loss cons: {},  val rmse: {}".format(i+1, loss_history['loss_sma'][-2], loss_history['loss_tml'][-2], loss_history['loss_cons'][-2], loss_history['val_rmse'][-1]))
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

	return X[lab], y[lab]   #, X[ulab], y[ulab] 


def run_train(target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, m_saveloc, r_saveloc, pretrain_loc=None, type_='r', alpha=0.1, model='EEGNET-LSTM', target_dataset='DDir', perc_data=1.0, perc_label=0.5, seed=0):
	n_epochs = int(np.max([100*perc_label, 20]))
	# n_epochs = 2
	# data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)
	# # target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_source()
	# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()

	# target_train_X, target_train_y, target_test_X, target_test_y = data_loader.get_data_source()
	# source_train_X, source_train_y, source_test_X, source_test_y = data_loader.get_data_target()
	
	pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
	
	np.random.seed(seed)

	# target_train_X_lab, target_train_y_lab, target_train_X_ulab, target_train_y_ulab = return_labeled_subets(target_train_X, target_train_y, perc_label) 
	target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)
	# del target_train_X, target_train_y
	# print(target_train_X_lab.shape, target_train_y_lab.shape, target_train_X_ulab.shape, target_train_y_ulab.shape, np.mean(target_train_y_lab), np.std(target_train_y_lab), np.mean(target_train_y_ulab), np.std(target_train_y_ulab)) 
	# sys.exit()

	obj_cdist = bbcn_eye(type_=type_, model_name=model, alpha=alpha, target_dataset=target_dataset)
	obj_cdist.make_model(pretrain_loc)
	obj_cdist.make_feat_extr_layer()
	
	start_time = datetime.now()
	loss_history = obj_cdist.train_model(target_train_X_lab, target_train_y_lab, target_train_X, target_train_y, target_val_X, target_val_y, target_test_y, n_epochs, pdetails, m_saveloc)
	end_time = datetime.now()
	print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
	# obj_cdist.save_model(m_saveloc)
	plot_curves(loss_history['loss_sma'], "SMA loss", "loss", n_epochs, r_saveloc+"/regression_target.png", x_label='Epoch')
	plot_curves(loss_history['loss_tml'], "TML loss", "loss", n_epochs, r_saveloc+"/total_target_loss.png", x_label='Epoch')
	plot_curves(loss_history['loss_cons'], "Cons loss", "loss", n_epochs, r_saveloc+"/total_target_loss.png", x_label='Epoch')
	plot_curves(loss_history['val_rmse'], "Validation Error", "Error", n_epochs, r_saveloc+"/val_rmse.png", x_label='Epoch')
	
	pdetails.close()
	del target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm
	del obj_cdist

def run_test(m_saveloc, r_saveloc, model='EEGNET-LSTM', type_='r', in_='pix', min_1=0, max_1=725, min_2=0, max_2=825, target_dataset='DDir', perc_data=1.0):
	
	data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)
	target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_source()
	# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()

	# target_train_X, target_train_y, target_test_X, target_test_y = data_loader.get_data_source()
	# source_train_X, source_train_y, source_test_X, source_test_y = data_loader.get_data_target()

	obj_cdist = bbcn_eye(type_=type_, model_name=model, target_dataset=target_dataset, p_y_gmm=p_y_gmm)
	obj_cdist.load_model(m_saveloc)
	

	####################################################target################################################################################

	eye_train = data_loader.inv_scale(target_train_y)
	eye_test = data_loader.inv_scale(target_test_y)
	eye_val = data_loader.inv_scale(target_val_y)

	pred_train_y = data_loader.inv_scale(obj_cdist.model_target.predict(target_train_X))
	pred_test_y = data_loader.inv_scale(obj_cdist.model_target.predict(target_test_X))
	pred_val_y = data_loader.inv_scale(obj_cdist.model_target.predict(target_val_X))
	#x-predictions
	
	
	fptr = open(r_saveloc+"performance_predictions.txt", 'w')
	if type_ == 'r':
		plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
		plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
		plot_predictions(pred_val_y[:, 0], eye_val[:, 0], r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1) 
		
		plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= "CRAFT (Base Model: VS)", addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
		print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y), get_rmse1(eye_val, pred_val_y)), file=fptr)
	elif type_ == 'theta':
		plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		plot_predictions(pred_val_y[:, 0], eye_val[:, 0], r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_angle(eye_train, pred_train_y), get_angle(eye_test, pred_test_y),  get_angle(eye_val, pred_val_y)), file=fptr)

	###################################################souce######################################################################################
	# eye_train = data_loader.inv_scale(source_train_y)
	# eye_test = data_loader.inv_scale(source_test_y)

	# pred_train_y = data_loader.inv_scale(obj_cdist.model_source.predict(source_train_X))
	# pred_test_y = data_loader.inv_scale(obj_cdist.model_source.predict(source_test_X))
	
	# #x-predictions
	# if type_ == 'r':
	#   plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_source.png", title_="Train (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
	#   plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_source.png", title_="Test (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
	#   print("Train RMSE (source): {}  \nTest RMSE (source): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y)), file=fptr)
	# elif type_ == 'theta':
	#   plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_source.png", title_="Train (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2, circ=True)
	#   plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_source.png", title_="Test (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2, circ=True)
	#   print("Train RMSE (source): {}  \nTest RMSE (source): {}".format(get_angle(eye_train, pred_train_y), get_angle(eye_test, pred_test_y)), file=fptr)
	fptr.close()
	del target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm
	del obj_cdist

def run_test_paper(data_loader, target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, m_saveloc, r_saveloc, type_='r', in_='pix', min_1=None, max_1=None, min_2=None, max_2=None, model='EEGNET-LSTM', target_dataset='DDir', target_plot_title=""):
	
	# data_loader = integrated_data(type_, target_dataset)
	# # target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_source()
	# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_target()

	obj_cdist = bbcn_eye(type_=type_)
	obj_cdist.load_model(m_saveloc)
	

	####################################################target################################################################################

	eye_train = data_loader.inv_scale(target_train_y)
	eye_test = data_loader.inv_scale(target_test_y)
	eye_val = data_loader.inv_scale(target_val_y)
	fptr = open(r_saveloc+"performance_predictions.txt", 'w')

	pred_train_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_train_X))

	start_time = datetime.now()
	pred_test_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_test_X))
	end_time = datetime.now()
	print("--- {} examples: {} minutes ---".format(target_test_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=fptr)
	pred_val_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_val_X))

	print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y), get_rmse1(eye_val, pred_val_y)), file=fptr)
	fptr.close()
	#x-predictions
	# plt.rcParams['text.usetex'] = True
	plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper_2.png", title_= target_plot_title, addon="$|\\Delta\\vec{{r}}|$ (in {})".format(in_), _min_use=min_2, _max_use=max_2)
	
if __name__ == "__main__":
	
	import gc
	
	base_mloc = ["./Models/rebuttal_final/bbcn_perc/"]
	base_rloc = ["./Results/rebuttal_final/bbcn_perc/"]
	
	dat_type = ["DDir"]
	model_names = ['EEGNET-LSTM']
	pretrain_loc_list = ["./Models/iclr/EEGNET-LSTM/source_VS/"]
	descriptions = ["EEGNET-LSTM: Prog-Mixup"]
	
	data_loader = integrated_data('r', dat_type[0], perc_data_target=1.0)
	target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()
	for m in range(len(base_mloc)):
		for i in [1, 2, 5, 10]:
			if not os.path.exists(base_rloc[m]+str(i)):
				os.mkdir(base_rloc[m]+str(i))
			if not os.path.exists(base_mloc[m]+str(i)):
				os.mkdir(base_mloc[m]+str(i))
			for j in range(3):
				if not os.path.exists(base_rloc[m]+str(i)+"/"+str(j)):
					os.mkdir(base_rloc[m]+str(i)+"/"+str(j))
				if not os.path.exists(base_mloc[m]+str(i)+"/"+str(j)):
					os.mkdir(base_mloc[m]+str(i)+"/"+str(j))
				run_train(target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, base_mloc[m]+str(i)+"/"+str(j)+"/", base_rloc[m]+str(i)+"/"+str(j)+"/", model=model_names[m], pretrain_loc=pretrain_loc_list[m], target_dataset=dat_type[m], type_='r', alpha=0.1, perc_data=1.0, perc_label=(0.01*i), seed=j)
				run_test_paper(data_loader, target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, base_mloc[m]+str(i)+"/"+str(j)+"/", base_rloc[m]+str(i)+"/"+str(j)+"/", model=model_names[m], type_='r', in_='pix', target_dataset=dat_type[m], target_plot_title=descriptions[m])
				# sys.exit()
				# break
				gc.collect()
				tf.keras.backend.clear_session()