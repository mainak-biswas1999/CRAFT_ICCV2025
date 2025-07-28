import os
import sys
import numpy as np
import pandas as pd
import re
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from loader import *
from data_struct import *
from sfcn_tensorflow import *
from plotter import *
from plot_paper import *

matplotlib.use('Agg') 

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# 	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
# 	try:
# 		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]) # Notice here
# 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 	except RuntimeError as e:
# 		# Virtual devices must be set before GPUs have been initialized
# 		print(e)

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

def predict(model, X):
	all_ops = []
	bs = 6
	n_times = int(np.ceil(X.shape[0]/bs))
	
	for i in range(n_times):
		if i == n_times - 1:
			end_pos = X.shape[0]
		else:
			end_pos = (i+1)*bs
		
		all_ops.append(model(X[i*bs : end_pos]).numpy())
	return np.concatenate(all_ops)

def run_emsemble(base_mloc_r, r_saveloc, addon=""):
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/age_scaling_uniform.npy"
	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")

	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)
	
	predictions_test = []
	actual_test = []
	for j in range(5):	
		#return
		obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
		base_mloc = base_mloc_r + str(j) + "/"
		
		tlsa_act = []
		tlsa_pred = []
		for i in range(n_fold):
			model = load_model(base_mloc+"fold_"+str(i+1)+"/"+addon)
			model.compile()
			model.summary()
			_, _, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
			tlsa_pred.append(inv_prep(predict(model, X_test_foldi), age_range_loc))
			tlsa_act.append(inv_prep(y_test_foldi, age_range_loc))
			del model
				
		actual_test.append(np.concatenate(tlsa_act))
		predictions_test.append(np.concatenate(tlsa_pred))


	predictions_test = np.concatenate(predictions_test, axis=1)
	actual_test = actual_test[0]
	print(predictions_test.shape, actual_test.shape)
	#checking if the order is ok -- sanity
	# print(np.sum(actual_test[:, 0] == actual_test[:, 1]), np.sum(actual_test[:, 1] == actual_test[:, 2]), np.sum(actual_test[:, 2] == actual_test[:, 3]), np.sum(actual_test[:, 3] == actual_test[:, 0]))
	
	sel_indices = np.array([[0,1,2,3], [0,1,2,4], [0,1,3,4], [0,2,3,4], [1,2,3,4]])
	for i in range(5):
		if not os.path.exists(r_saveloc+"ensemble/"+str(i)):
			os.makedirs(r_saveloc+"ensemble/"+str(i))
		
		pred_test_y = np.mean(predictions_test[:, sel_indices[i]], axis=1).ravel()
		plot_predictions_paper(pred_test_y, actual_test, r_saveloc+"ensemble/"+str(i)+"/overall_results_paper_2.png", title_="SFCN CRAFT: Ensemble", addon="Age (in years)")

if __name__ == "__main__":
	run_emsemble("./Models/iclr/cdist_0.1_subsample_0.33_radcModel/", "./Results/iclr/cdist_0.1_subsample_0.33_radcModel/", addon="Decoder_cdist")