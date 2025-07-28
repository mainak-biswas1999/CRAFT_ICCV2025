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

def plot_predictions_paper_wo_density(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
	y_age = y = np.squeeze(y)
	pred_y_age = pred_y = np.squeeze(pred_y)
	
	#y_age = scale_obj.inv_scale(y)
	#pred_y_age = scale_obj.inv_scale(pred_y)
	
	_min = np.min([np.min(y_age), np.min(pred_y_age)]) 
	_max = np.max([np.max(y_age), np.max(pred_y_age)])
	
	if _min_use is None and _max_use is None:
		x_min_use, x_max_use = _min - 1, _max + 1
		y_min_use, y_max_use = _min - 1, _max + 1
	elif _min_use is not None and _max_use is None:
		x_min_use, x_max_use = _min_use, _max + 1
		y_min_use, y_max_use = _min_use, _max + 1
	elif _min_use is None and _max_use is not None:
		x_min_use, x_max_use = _min - 1, _max_use
		y_min_use, y_max_use = _min - 1, _max_use
	else:
		x_min_use, x_max_use = _min_use, _max_use
		y_min_use, y_max_use = _min_use, _max_use

	#generate the plot
	fig, ax = plt.subplots(figsize=(12, 12))
	
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(5)
	ax.spines['left'].set_linewidth(5)

	
	plt.xlabel("Chronological "+addon, fontsize=45, fontname="Myriad Pro")
	plt.ylabel("Predicted (Brain Age) "+addon, fontsize=45, fontname="Myriad Pro")
	plt.title(title_, fontsize=55, fontname="Myriad Pro")
	plt.xlim(x_min_use, x_max_use)
	plt.ylim(y_min_use, y_max_use)

	mX, mY = np.mgrid[x_min_use:x_max_use:2000j, y_min_use:y_max_use:2000j]

	
	
	# mesh = np.vstack([mX.ravel(), mY.ravel()])
	# data = np.vstack([y_age, pred_y_age])
	# kde = gaussian_kde(data)
	# density = kde(mesh).reshape(mX.shape)
	
	mesh = np.hstack([np.expand_dims(mX.ravel(), axis=-1), np.expand_dims(mY.ravel(), axis=-1)])
	data = np.hstack([np.expand_dims(y_age, axis=-1), np.expand_dims(pred_y_age, axis=-1)])
	p_y_gmm = GaussianMixture(n_components=4, random_state=0).fit(data)
	# print(data.shape, mesh.shape, p_y_gmm.score_samples(mesh).shape)
	density = np.exp(p_y_gmm.score_samples(mesh).reshape(mX.shape))
	
	alpha = 1.5
	density2 = (-1*p_y_gmm.score_samples(data))   #**0.000005
	#density2 = ((density2 - np.min(density2))/(np.max(density2) - np.min(density2)))**0.4
	density2 = (density2 - np.mean(density2))*alpha

	density2 = 1/(1 + np.exp(-1*density2))
	print(np.min(density2), np.max(density2), np.mean(density2))

	# plt.scatter(y_age, pred_y_age, c=density2, cmap="hot", alpha=0.75, s=400, edgecolor='black', linewidth=3) #color='k', edgecolor='black')
	plt.scatter(y_age, pred_y_age, alpha=0.9, s=400, edgecolor='black', linewidth=2, color='w', marker='^')

	
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=40)
	plt.rc('xtick', labelsize=40)
	plt.rc('ytick', labelsize=40)
	plt.locator_params(axis='both', nbins=6) 
	ax.tick_params(width=8)
	
	yex = np.linspace(_min - 1, _max + 1, 10000)
	if circ == False:
		corr = png.corr(np.squeeze(y_age), np.squeeze(pred_y_age), method='percbend')
		mse_error = np.round(np.sqrt(np.mean((y_age - pred_y_age)**2)), 2)
		mae_error = np.round(np.mean(np.abs((y_age - pred_y_age))), 2)
		#print(mse_error, mae_error)
		# ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}, mae={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error, mae_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
		ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
		################################print best fit#############################################
		A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
		w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
			
		y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
		# plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='plum')
		plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='indianred')
	else:
		corr, pval = png.circ_corrcc(np.squeeze(y_age), np.squeeze(pred_y_age), correction_uniform=True)
		mse_error = get_angle(np.squeeze(y_age), np.squeeze(pred_y_age))
		#print(mse_error, mae_error)
		ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}".format(str(np.round(np.abs(corr), 3)), np.maximum(np.round(pval, 3), 0.001), str(np.round(mse_error, 2))), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
	
	################################print y=x##################################################
	plt.plot(yex, yex, linestyle = 'dashed', linewidth=8, zorder=8, color='grey')
	
		
	#plt.title("r= {}, p= {}".format(np.round(corr['r'][0], 2), np.round(corr['p-val'][0], 3)))
	plt.savefig(saveloc, dpi=300, bbox_inches="tight")
	plt.close()



def disp_results(base_rloc, base_mloc):
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"


	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

	#print(y_train_tlsa)
	#return
	obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
	# pdetails = open(base_rloc+"exec_time.txt", 'w')
	tlsa_act = []
	tlsa_pred = []
	
	for i in range(n_fold):
		#make places to save
		#train the model
		X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
		# print(wrongly_sample_index.shape)
		tlsa_act.append(y_test_foldi)
		obj_model = SFCN_tf()
		obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/Decoder_cdist/")

		#plot training curves
		pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
		pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
		del obj_model
		
		print(pred_y_train_foldi.shape, pred_y_test_foldi.shape)
		tlsa_pred.append(pred_y_test_foldi)

	tlsa_act = np.concatenate(tlsa_act)
	tlsa_pred = np.concatenate(tlsa_pred)
	plot_predictions_paper_wo_density(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper_square_paper3.png", title_="SFCN: CRAFT", addon="Age")
	
if __name__ == '__main__':
	disp_results("./Results/rebuttal/craft_perc/60/0/", "./Models/rebuttal/craft_perc/60/0/")