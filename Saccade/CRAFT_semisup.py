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
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda, UpSampling1D, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
from datetime import datetime

matplotlib.use('Agg') 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]) # Notice here
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)

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

class EM_Exp_GMM_1d:
	def __init__(self, n_g, n_e, X):
		self.n_g = n_g	#number of gaussians = k1
		self.n_e = n_e	#number of exponentials = k2
		
		self.X = X		#(N, 1)
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
		self.betas = None	#k2 exponentials
		
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
		norm_term = np.expand_dims(self.X.shape[0] * self.alphas, axis=1)	#converting it to k1x1
		self.mus = (self.q1.T @ self.X) / norm_term
		#sigmas = np.zeros(self.sigmas.shape)
		for i in range(self.n_g):
			#X has examples in the rows
			mu_i = np.expand_dims(self.mus[i, 0], axis=0)
			#\|q (X-u)
			X_tilde = (self.X - mu_i) * np.sqrt(np.expand_dims(self.q1[:, i], axis=1))
			self.sigmas[i, 0] = np.sqrt((X_tilde.T @ X_tilde)[0, 0] / norm_term[i, 0]) 
		#exponentials - 
		norm_term2 = np.expand_dims(self.X.shape[0] * self.betas, axis=1)	#converting it to k2x1
		self.lambdas = norm_term2 / (self.q2.T @ self.X) 
		 

	def run_EM(self, max_iter=100):
		for i in range(max_iter):
			self.Expectation()
			self.Maximization()
	
	def return_P(self, X):
		P_normal = self.get_all_normal_values(X)
		P_exp = self.get_all_exp_values(X)
		P =	P_exp @ np.expand_dims(self.betas, axis=1) +	P_normal @ np.expand_dims(self.alphas, axis=1)
		return P

class CRAFT(object):
	def __init__(self, type_='r', model_name='EEGNET-LSTM', output_shape=1, alpha=1.0, bs=128, lr=0.0001, n_bins=800, target_dataset='DDir', p_y_gmm=None):
									 # L0,	L1,	L2, L3, op
		self.target_dataset = target_dataset
		self.p_y_gmm = p_y_gmm
		self.alpha = alpha
		self.lr = lr
		self.bs = bs
		self.output_shape = output_shape
		self.model_source = None		#classification based on features - C o F for the source
		self.model_name = model_name
		self.N_total = None
		
		self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
		
		self.range_target = None
		self.target_labels = None
		self.exp_diff_prior = None
		self.n_bins = n_bins
		self.type_	= type_
	
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
		self.model_source.save(saveloc+"Efects_Decoder_source")
		
	def load_model(self, saveloc):
		self.model_source = load_model(saveloc+"Efects_Decoder_source", compile=False)
		# self.model_source = load_model(saveloc, compile=False)
		self.model_source.compile()
		self.model_source.summary()
	
	def angle_loss(self, y_true, y_pred):
		# return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(y_pred - y_true), tf.cos(y_pred - y_true)))))
		return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(y_pred - y_true)), tf.cos(np.pi*(y_pred - y_true))))))

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
		
	def get_pseudo_labels_bins(self, target_X):
		target_pred = self.model_source(target_X, training=False)
		if self.type_ == 'r':
			exp_diff_posterior = tf.cast(tf.math.exp(-(target_pred - self.age_labels)**2), 'float32')
		else:
			exp_diff_posterior = tf.cast(tf.math.exp(-tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(target_pred - self.age_labels)), tf.cos(np.pi*(target_pred - self.age_labels)))))), 'float32')

		cdist_rescaled_posteriors = (exp_diff_posterior * self.exp_diff_prior)/(tf.math.reduce_sum(exp_diff_posterior, axis=0, keepdims=True))
		pos_max = tf.cast(tf.math.argmax(cdist_rescaled_posteriors, axis=1), 'float32')
		psuedo_labels = tf.expand_dims(self.range_target[0] + (pos_max + 0.5) * (self.range_target[1] - self.range_target[0]) / self.n_bins, axis=1)
		
		return psuedo_labels

	def loss_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=False):
		pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
		pred_y_t_ulab = self.model_source(batch_target_ulab, training=to_train)
		psuedo_labels = self.get_pseudo_labels_bins(batch_target_ulab)
		#uncomment this
		if self.type_ == 'r':
			loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
			loss_t_on_source = MeanSquaredError()(psuedo_labels, pred_y_t_ulab)
			pred_y_t_regu = tf.repeat(tf.transpose(pred_y_t_ulab), repeats=[batch_y_target_ulab.shape[0]], axis=0)
			#mse
			# regu_diff = tf.math.exp(-(pred_y_t_regu - psuedo_labels)**2)
			regu_diff = -(pred_y_t_regu - psuedo_labels)**2
		else:
			loss_regression_t = self.angle_loss(batch_y_target_lab, pred_y_t_lab)
			loss_t_on_source = self.angle_loss(psuedo_labels, pred_y_t_ulab)
			pred_y_t_regu = tf.repeat(tf.transpose(pred_y_t_ulab), repeats=[batch_y_target_ulab.shape[0]], axis=0)
			#theta
			regu_diff = tf.math.exp(-tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(pred_y_t_regu - psuedo_labels)), tf.cos(np.pi*(pred_y_t_regu - psuedo_labels))))))
		
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

	def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_val_X, target_val_y, target_test_y, num_epochs, printDetails, m_saveloc):
		
		# self.range_target = [np.min(y_target_lab), np.max(y_target)]
		self.range_target = [-1.0, 1.0]
		create_labels = (np.expand_dims(np.linspace(self.range_target[0], self.range_target[1], self.n_bins), axis=0) + 0.5*(self.range_target[1] - self.range_target[0])/(self.n_bins - 1)).astype('float32')
		# print(create_labels.shape)
		if self.type_ == 'theta': 
			p_y_gmm = GaussianMixture(n_components=11, random_state=0).fit(y_target_lab)
			densities_py = np.exp(np.expand_dims(p_y_gmm.score_samples(create_labels.T), axis=0))

			self.plot_histo(create_labels.ravel(), densities_py.ravel(), "theta (in rads) - scaled", "Density", "del theta", "./Results/iclr/stats/theta_ddir_train_fit_gmm_-1_1_lg.png")
		
		elif self.type_ == 'r':
			if self.target_dataset == 'DDir':
				if self.p_y_gmm is None:
					p_y_gmm = GaussianMixture(n_components=14, random_state=0).fit(y_target_lab)	#np.concatenate([y_target, target_val_y], axis=0))
				
				densities_py = np.exp(np.expand_dims(self.p_y_gmm.score_samples(create_labels.T), axis=0))
				self.plot_histo(create_labels.ravel(), densities_py.ravel(), "r (in pix) - scaled", "Density", "del r", "./Results/iclr/stats/r_ddir_train_fit_gmm_-1_1_lg.png")
			else:
				if self.p_y_gmm is None:	
					p_y_gmm = EM_Exp_GMM_1d(2, 2, y_target_lab+1)	 # from 0 onwards to make the exponential work
					p_y_gmm.run_EM()
				
				densities_py = self.p_y_gmm.return_P(create_labels.T + 1).T
				self.plot_histo(create_labels.ravel(), densities_py.ravel(), "r (in pix) - scaled", "Density", "del r", "./Results/iclr/stats/r_ddir_train_fit_gmm_-1_1_wm.png")
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
				
				if len(train_data_target_t) < self.bs:
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
				t_loss, loss_regression_t = self.back_prop_cdist(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab, batch_y_target_ulab)
				#keep track of the loss
				loss_history['loss_regression_t_total'].append(loss_regression_t.numpy())
				loss_history['loss_t'].append(t_loss.numpy())


			pred_val_y = self.model_source.predict(target_val_X)
			loss_history['val_rmse'].append(get_rmse1(target_val_y, pred_val_y))
			# if loss_history['val_rmse'][-1] < best_rmse:
			# 	best_rmse = loss_history['val_rmse'][-1]
			# 	self.save_model(m_saveloc)
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

	return X[lab], y[lab]	#, X[ulab], y[ulab] 

def run_train(target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, m_saveloc, r_saveloc, pretrain_loc=None, type_='r', alpha=1.0, model='EEGNET-LSTM', target_dataset='DDir', perc_data=1.0, perc_label=0.5, seed=0):
	n_epochs = int(np.max([100*perc_label, 20]))
	
	pdetails = open(r_saveloc+"cdist_learning.txt", 'w')
	
	np.random.seed(seed)

	target_train_X_lab, target_train_y_lab = return_labeled_subets(target_train_X, target_train_y, perc_label)
	

	obj_cdist = CRAFT(type_=type_, model_name=model, alpha=alpha, target_dataset=target_dataset, p_y_gmm=p_y_gmm)
	obj_cdist.make_model(pretrain_loc)
	
	start_time = datetime.now()
	loss_history = obj_cdist.train_model(target_train_X_lab, target_train_y_lab, target_train_X, target_train_y, target_val_X, target_val_y, target_test_y, n_epochs, pdetails, m_saveloc)
	end_time = datetime.now()
	print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, target_train_X.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
	# obj_cdist.save_model(m_saveloc)
	plot_curves(loss_history['loss_regression_t_total'], "Total Target Loss (incl. pseudoloss)", "loss", n_epochs, r_saveloc+"/regression_target.png", x_label='Epoch')
	plot_curves(loss_history['loss_t'], "Target Regression Loss", "loss", n_epochs, r_saveloc+"/total_target_loss.png", x_label='Epoch')
	plot_curves(loss_history['val_rmse'], "Validation Error", "Error", n_epochs, r_saveloc+"/val_rmse.png", x_label='Epoch')
	
	pdetails.close()
	del target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm
	del obj_cdist

def run_test(m_saveloc, r_saveloc, model='EEGNET-LSTM', type_='r', in_='pix', min_1=0, max_1=725, min_2=0, max_2=825, target_dataset='DDir', perc_data=1.0):
	
	data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)
	target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_source()
	

	obj_cdist = CRAFT(type_=type_, model_name=model, target_dataset=target_dataset, p_y_gmm=p_y_gmm)
	obj_cdist.load_model(m_saveloc)
	

	####################################################target################################################################################

	eye_train = data_loader.inv_scale(target_train_y)
	eye_test = data_loader.inv_scale(target_test_y)
	eye_val = data_loader.inv_scale(target_val_y)

	pred_train_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_train_X))
	pred_test_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_test_X))
	pred_val_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_val_X))
	#x-predictions
	
	
	fptr = open(r_saveloc+"performance_predictions.txt", 'w')
	if type_ == 'r':
		plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
		plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
		plot_predictions(pred_val_y[:, 0], eye_val[:, 0], r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)	
		
		plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= "CRAFT (Base Model: VS)", addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
		print("Train RMSE (Target): {}	\nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y), get_rmse1(eye_val, pred_val_y)), file=fptr)
	elif type_ == 'theta':
		plot_predictions(pred_train_y[:, 0], eye_train[:, 0], r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		plot_predictions(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		plot_predictions(pred_val_y[:, 0], eye_val[:, 0], r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
		print("Train RMSE (Target): {}	\nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_angle(eye_train, pred_train_y), get_angle(eye_test, pred_test_y),	get_angle(eye_val, pred_val_y)), file=fptr)

	
	fptr.close()
	del target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm
	del obj_cdist

def run_test_paper(target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, m_saveloc, r_saveloc, type_='r', in_='pix', min_1=None, max_1=None, min_2=None, max_2=None, model='EEGNET-LSTM', target_dataset='DDir', target_plot_title=""):
	
	# data_loader = integrated_data(type_, target_dataset)
	# # target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_source()
	# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_target()

	obj_cdist = CRAFT(type_=type_)
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

	print("Train RMSE (Target): {}	\nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y), get_rmse1(eye_val, pred_val_y)), file=fptr)
	fptr.close()
	#x-predictions
	# plt.rcParams['text.usetex'] = True
	plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper_2.png", title_= target_plot_title, addon="$|\\Delta\\vec{{r}}|$ (in {})".format(in_), _min_use=min_2, _max_use=max_2)
	
if __name__ == "__main__":
	
	import gc
	
	base_mloc = ["./Models/rebuttal/craft_perc_sup_sacc/", "./Models/rebuttal/craft_PyrCNN_perc_sup_sacc/", "./Models/rebuttal/craft_perc_sup_wm/", "./Models/rebuttal/craft_PyrCNN_perc_sup_wm/"]
	base_rloc = ["./Results/rebuttal/craft_perc_sup_sacc/", "./Results/rebuttal/craft_PyrCNN_perc_sup_sacc/", "./Results/rebuttal/craft_perc_sup_wm/", "./Results/rebuttal/craft_PyrCNN_perc_sup_wm/"]
	
	base_mloc_tl = ["./Models/rebuttal/perc_sup_sacc/", "./Models/rebuttal/PyrCNN_perc_sup_sacc/", "./Models/rebuttal/perc_sup_wm/", "./Models/rebuttal/PyrCNN_perc_sup_wm/"]
	dat_type = ["DDir", "DDir", "WM", "WM"]
	model_names = ['EEGNET-LSTM', 'PyrCNN', 'EEGNET-LSTM', 'PyrCNN']
	pretrain_loc_list = ["./Models/iclr/EEGNET-LSTM/source_VS/", "./Models/iclr/Spyr_CNN/source_VS/", "./Models/iclr/EEGNET-LSTM/source_VS/", "./Models/iclr/Spyr_CNN/source_VS/"]
	descriptions = ["EEGNET-LSTM: CRAFT", "PyrCNN: CRAFT", "EEGNET-LSTM: CRAFT", "PyrCNN: CRAFT"]
	
	# data_loader = integrated_data('r', dat_type[0], perc_data_target=1.0)
	# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()
	for m in [0, 2]:
		data_loader = integrated_data('r', dat_type[m], perc_data_target=1.0)
		target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()
		
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
				run_test_paper(target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm, base_mloc[m]+str(i)+"/"+str(j)+"/", base_rloc[m]+str(i)+"/"+str(j)+"/", model=model_names[m], type_='r', in_='pix', target_dataset=dat_type[m], target_plot_title=descriptions[m])
				gc.collect()
				tf.keras.backend.clear_session()