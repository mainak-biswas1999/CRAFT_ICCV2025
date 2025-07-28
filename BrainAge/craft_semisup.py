import os
import sys
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from loader import *
from data_struct import *
from sfcn_tensorflow import *
from plotter import *
from plot_paper import *
from datetime import datetime

matplotlib.use('Agg') 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24000)]) # Notice here
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)

def plot_histo(density_est_x, xlabel_="Age (in years)", y_label="Density", saveloc="./histo.png"):
		#plot the histogram of velocities
	
	plt.hist(density_est_x, bins=40, density=True)
	

	plt.xlabel(xlabel_, fontsize=16)
	plt.ylabel(y_label, fontsize=16)
	plt.savefig(saveloc, dpi=200, bbox_inches="tight")
	plt.close()

class CDIST_sfcn(object):
	def __init__(self, output_shape=1, alpha=0.1, bs=4, lr=0.0001, use_pseudo=True, n_bins = 400):
								   # L0,  L1,  L2, L3, op
		self.alpha = alpha
		self.lr = lr
		self.bs = bs
		self.output_shape = output_shape
		self.model_source = None		#classification based on features - C o F for the source
				
		self.optimizer_model_s = tf.keras.optimizers.Adam(learning_rate=self.lr)
		self.use_pseudo = use_pseudo
		
		if self.use_pseudo == True:
			self.target_mu = None
			self.target_sigma = None
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
		self.model_source.save(saveloc+"Decoder_cdist")
		
	def load_model(self, saveloc, if_cdist=True):
		if if_cdist== True:
			self.model_source = load_model(saveloc+"Decoder_cdist", compile=False)
		else:
			self.model_source = load_model(saveloc, compile=False)
		self.model_source.compile()
	
	def make_model(self):
		#get the predictor model
		#the feature extractor output
		mod_object = SFCN_tf(pretrained=True, to_train_full=True)
		self.model_source = mod_object.add_mse_head(True)
		del mod_object
		self.model_source.compile()
		self.model_source.summary()
	
	def predict_mse(self, X):
		all_ops = []
		n_times = int(np.ceil(X.shape[0]/self.bs))
		
		for i in range(n_times):
			if i == n_times - 1:
				end_pos = X.shape[0]
			else:
				end_pos = (i+1)*self.bs
			#print(i*self.hyperparameter['bs_pred'], end_pos)
			
			age_pred = self.model_source(X[i*self.bs : end_pos]).numpy()
			all_ops.append(age_pred.squeeze(axis=-1))
			
		return np.concatenate(all_ops)
	
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
		regularizer = self.alpha * (loss_t_on_source + tf.math.reduce_mean(sum_ydiff_for_eachX))

		t_loss = loss_regression_t + regularizer
		return t_loss, loss_regression_t

	def loss_cdist_sup(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=False):
		
		pred_y_t_lab = self.model_source(batch_target_lab, training=to_train)
		loss_regression_t = MeanSquaredError()(batch_y_target_lab, pred_y_t_lab)
		
		return loss_regression_t

	def loss_cdist_usup(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=False):
		pred_y_t_ulab = self.model_source(batch_target_ulab, training=to_train)
		psuedo_labels = self.get_pseudo_labels_bins(batch_target_ulab)
		#uncomment this
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
		regularizer = self.alpha * (loss_t_on_source + tf.math.reduce_mean(sum_ydiff_for_eachX))

		return regularizer
	

	# def back_prop_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab):
		
	# 	with tf.GradientTape(persistent=True) as tape:
	# 		#classification loss
	# 		t_loss, loss_regression_t = self.loss_cdist(batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=True)
			
	# 	#update the classifier gradients
	# 	grad_class = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
	# 	self.optimizer_model_s.apply_gradients(zip(grad_class, self.model_source.trainable_variables))
	# 	del grad_class

	# 	#grad_class = tape.gradient(regularizer, self.model_source.trainable_variables)
	# 	#self.optimizer_model_s.apply_gradients(zip(grad_class, self.model_source.trainable_variables))
		
	# 	return t_loss, loss_regression_t

	def back_prop_cdist(self, batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab):
		
		with tf.GradientTape() as tape:
			#classification loss
			loss_regression_t = self.loss_cdist_sup(batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=True)
			
		grad_loss_regression = tape.gradient(loss_regression_t, self.model_source.trainable_variables)
		#self.optimizer_model_s.apply_gradients(zip(grad_class, self.model_source.trainable_variables))
		#del grad_class

		with tf.GradientTape() as tape:
			#classification loss
			regularizer = self.loss_cdist_usup(batch_target_lab, batch_y_target_lab, batch_target_ulab, batch_y_target_ulab, to_train=True)
			
		#update the classifier gradients
		grad_regularizer = tape.gradient(regularizer, self.model_source.trainable_variables)
		total_grad = []
		for i in range(len(grad_loss_regression)):
			total_grad.append(grad_loss_regression[i] + grad_regularizer[i])
		# gradient step with the accumulated gradients
		self.optimizer_model_s.apply_gradients(zip(total_grad, self.model_source.trainable_variables))
		
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

	# def train_model(self, X_target, y_target, y_gmm_data, num_epochs, printDetails):
	def train_model(self, X_target_lab, y_target_lab, X_target_ulab, y_target_ulab, target_train_y, num_epochs, printDetails, m_saveloc):
		plot_histo(y_target_ulab.ravel(), xlabel_="Age (in years)", y_label="Density", saveloc="./Results/"+m_saveloc[9:]+"age_distr_train.png")
		if self.use_pseudo == True:
			#GMM
			self.range_target = [0., 1.]
			create_labels = (np.expand_dims(np.linspace(self.range_target[0], self.range_target[1], self.n_bins), axis=0) + 0.5*(self.range_target[1] - self.range_target[0])/(self.n_bins - 1)).astype('float32')
			p_y_gmm = GaussianMixture(n_components=1, random_state=0).fit(target_train_y)
			densities_py = np.exp(np.expand_dims(p_y_gmm.score_samples(create_labels.T), axis=0))
			
			self.plot_histo(create_labels.ravel(), densities_py.ravel(), "r (in pix) - scaled", "Density", "del r", "./Results/"+m_saveloc[9:]+"fit_age_tlsa_0_1.png")
			# sys.exit()
			self.age_labels = tf.constant(create_labels)
			self.exp_diff_prior = tf.cast(densities_py, 'float32')

		loss_history = {
						'loss_regression_t': [],
						'loss_t': []
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

				t_loss, loss_regression_t = self.back_prop_cdist(batch_X_target_lab, batch_y_target_lab, batch_X_target_ulab, batch_y_target_ulab)
				#keep track of the loss
				loss_history['loss_regression_t'].append(loss_regression_t.numpy())
				loss_history['loss_t'].append(t_loss.numpy())
				#print(loss_history['loss_regression_t'], loss_history['loss_t'])	
			if i%2 == 0:
				#import pdb;pdb.set_trace()
				print("Losses at epoch {}, Total T loss: {}, Target Reg. Loss: {}".format(i+1, loss_history['loss_t'][-2], loss_history['loss_regression_t'][-2]), file=printDetails, flush=True)
				print("Losses at epoch {}, Total T loss: {}, Target Reg. Loss: {}".format(i+1, loss_history['loss_t'][-2], loss_history['loss_regression_t'][-2]))
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


def caller_paper_subsamp_kseed(base_rloc_r, base_mloc_r, perc_label=0.2, pretrain_loc=None, alpha=0.1, bin_size=400, check_train=False):
	n_epochs = 15		#keep the no of updates fixed
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"
	# if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
	# 	os.mkdir(base_rloc_r+str(int(perc_label*100)))
	# if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
	# 	os.mkdir(base_mloc_r+str(int(perc_label*100)))
	# base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
	# base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"
	if not os.path.exists(base_rloc_r+str(bin_size)):
		os.mkdir(base_rloc_r+str(bin_size))
	if not os.path.exists(base_mloc_r+str(bin_size)):
		os.mkdir(base_mloc_r+str(bin_size))
	base_mloc_r = base_mloc_r+str(bin_size)+"/"
	base_rloc_r = base_rloc_r+str(bin_size)+"/"


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
		# pdetails = open(base_rloc+"exec_time.txt", 'w')
		tlsa_act = []
		tlsa_pred = []
		start_time = datetime.now()

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
			obj_model = CDIST_sfcn(alpha=alpha, n_bins=bin_size)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				# obj_model.make_model()
				# obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/", True)
				obj_model.load_model(pretrain_loc+str(j)+"/fold_"+str(i+1)+"/", False)
				loss_history = obj_model.train_model(X_train_foldi_sub_samp, y_train_foldi_sub_samp, X_train_foldi, y_train_foldi, y_train_foldi, n_epochs, pdetails, base_mloc)
				obj_model.save_model(base_mloc+"fold_"+str(i+1)+"/")
				plot_curves(loss_history['loss_regression_t'], "Target Regression Loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/regression_targetloss.png", x_label='Epoch')
				plot_curves(loss_history['loss_t'], "Total Target Loss (incl. CDIST)", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/regression_source.png", x_label='Epoch')
			
			#plot training curves
			pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
			pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
			del obj_model
			
			print(pred_y_train_foldi.shape, pred_y_test_foldi.shape)
			tlsa_pred.append(pred_y_test_foldi)
			
			plot_predictions(pred_y_train_foldi, inv_prep(y_train_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/train_tlsa.png", title_="Train: SFCN finetune", addon= "Age (in years)")
			plot_predictions(pred_y_test_foldi, inv_prep(y_test_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/test_tlsa.png", title_="Test: SFCN finetune", addon= "Age (in years)")
		end_time = datetime.now()
		

		tlsa_act = np.concatenate(tlsa_act)
		tlsa_pred = np.concatenate(tlsa_pred)
		print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, tlsa_act.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
		plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper_square.png", title_="SFCN: CRAFT", addon="Age")
		
		pdetails.close()
		# return


def caller_paper_wrong_samp_kseed(base_rloc_r, base_mloc_r, perc_label=0.2, pretrain_loc=None, alpha=0.1, bin_size=400, check_train=False):
	n_epochs = 15		#keep the no of updates fixed
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"
	# if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
	# 	os.mkdir(base_rloc_r+str(int(perc_label*100)))
	# if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
	# 	os.mkdir(base_mloc_r+str(int(perc_label*100)))
	# base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
	# base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"
	# if not os.path.exists(base_rloc_r+str(bin_size)):
	# 	os.mkdir(base_rloc_r+str(bin_size))
	# if not os.path.exists(base_mloc_r+str(bin_size)):
	# 	os.mkdir(base_mloc_r+str(bin_size))
	base_mloc_r = base_mloc_r  #+str(bin_size)+"/"
	base_rloc_r = base_rloc_r  #+str(bin_size)+"/"


	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

	#print(y_train_tlsa)
	#return
	for j in [0, 2]:
		obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
		#return
		base_rloc = base_rloc_r + str(j) + "/"
		base_mloc = base_mloc_r + str(j) + "/"
		if not os.path.exists(base_rloc):
			os.mkdir(base_rloc)
		if not os.path.exists(base_mloc):
			os.mkdir(base_mloc)
		 
		pdetails = open(base_rloc+"self_kfold_tlsa.txt", 'w')
		# pdetails = open(base_rloc+"exec_time.txt", 'w')
		tlsa_act = []
		tlsa_pred = []
		start_time = datetime.now()

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

			wrongly_sample_index = np.where(y_train_foldi>0.5)[0]
			subselect_data = wrongly_sample_index[np.random.choice(len(wrongly_sample_index), int(0.20*len(wrongly_sample_index)), replace=False)]
			selected_index = np.concatenate([np.where(y_train_foldi<0.5)[0], subselect_data])
			X_train_foldi_wrong_samp = X_train_foldi[selected_index]
			y_train_foldi_wrong_samp = y_train_foldi[selected_index]
			
			X_train_foldi_sub_samp, y_train_foldi_sub_samp, _, _ = return_labeled_subets(X_train_foldi_wrong_samp, y_train_foldi_wrong_samp, perc_label)

			tlsa_act.append(y_test_foldi)
			obj_model = CDIST_sfcn(alpha=alpha, n_bins=bin_size)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				# obj_model.make_model()
				# obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/", True)
				obj_model.load_model(pretrain_loc+str(j)+"/fold_"+str(i+1)+"/", False)
				loss_history = obj_model.train_model(X_train_foldi_sub_samp, y_train_foldi_sub_samp, X_train_foldi_wrong_samp, y_train_foldi_wrong_samp, y_train_foldi, n_epochs, pdetails, base_mloc+"fold_"+str(i+1)+"/")
				obj_model.save_model(base_mloc+"fold_"+str(i+1)+"/")
				plot_curves(loss_history['loss_regression_t'], "Target Regression Loss", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/regression_targetloss.png", x_label='Epoch')
				plot_curves(loss_history['loss_t'], "Total Target Loss (incl. CDIST)", "loss", n_epochs, base_rloc+"fold_"+str(i+1)+"/regression_source.png", x_label='Epoch')
			
			#plot training curves
			pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
			pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
			del obj_model
			
			print(pred_y_train_foldi.shape, pred_y_test_foldi.shape)
			tlsa_pred.append(pred_y_test_foldi)
			
			plot_predictions(pred_y_train_foldi, inv_prep(y_train_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/train_tlsa.png", title_="Train: SFCN finetune", addon= "Age (in years)")
			plot_predictions(pred_y_test_foldi, inv_prep(y_test_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/test_tlsa.png", title_="Test: SFCN finetune", addon= "Age (in years)")
		end_time = datetime.now()
		

		tlsa_act = np.concatenate(tlsa_act)
		tlsa_pred = np.concatenate(tlsa_pred)
		print("--- {} epochs, {} labels, {} minutes ---".format(n_epochs, tlsa_act.shape[0], (end_time - start_time).total_seconds() / 60.0), file=pdetails)
		plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper_square.png", title_="SFCN: CRAFT", addon="Age")
		
		pdetails.close()
		# return


if __name__ == '__main__':
	perc_labels = [0.2, 0.4, 0.6]
	for perc_label in perc_labels:
		caller_paper_subsamp_kseed("./Results/rebuttal/craft_perc/", "./Models/rebuttal/craft_perc/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", check_train=False)
	# perc_label = 0.4
	# # bin_sizes = [200, 300, 500]
	# bin_sizes = [200, 500]
	# for bin_size in bin_sizes:
	# 	caller_paper_subsamp_kseed("./Results/hyperparameters/bin_size/40/", "./Models/hyperparameters/bin_size/40/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", alpha=0.1, bin_size=bin_size, check_train=False)
	# caller_paper_wrong_samp_kseed("./Results/sampling_errors_50_perc/craft/", "./Models/sampling_errors_50_perc/craft/", perc_label=perc_label, pretrain_loc="./Models/sampling_errors_50_perc/tl/", alpha=0.1, check_train=False)