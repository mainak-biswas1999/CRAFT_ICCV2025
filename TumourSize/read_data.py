import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import tqdm


def mse_loss(x, y):
	x = np.squeeze(x)
	y = np.squeeze(y)
	mse_error = np.round(np.sqrt(np.mean((y - x)**2)), 2)
	return mse_error

def read_cam(img_loc, sub_samp=False):
	dat = np.load(img_loc)

	X, y = dat['X'], np.expand_dims(dat['Y'], axis=-1)
	if sub_samp == True:
		locs = np.where((np.squeeze(y)>0.2) * (np.squeeze(y)<0.8))[0]
		X = X[locs]
		y = y[locs]

	train_test = k_fold_generator(np.squeeze(y), 5)
	X_train, y_train, X_test, y_test = train_test.get_kth_train_test_split2(X, y, 0)

	train_val = k_fold_generator(np.squeeze(y_train), 5)
	X_train, y_train, X_val, y_val = train_val.get_kth_train_test_split2(X_train, y_train, 0)

	print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
	return X_train, y_train, X_val, y_val, X_test, y_test


def load_id_ood_cam(log_scale=False):
	# id 
	id_ = np.load('./data/cam16.npz')
	# X_id, y_id = id_['X'].astype('float32')/255.0, np.expand_dims(id_['Y'], axis=-1)
	X_id, y_id = id_['X'], np.expand_dims(id_['Y'], axis=-1)
	# ood
	# X_ood, y_ood = read_cam('./data/cam17/')
	ood = np.load('./data/cam17.npz')
	X_ood, y_ood = ood['X'], np.expand_dims(ood['Y'], axis=-1)
	# locs = np.where((np.squeeze(y_ood)>0.1) * (np.squeeze(y_ood)<0.9))[0]
	# X_ood = X_ood[locs]
	# y_ood = y_ood[locs]

	

	print("OOD details: ", X_ood.shape, y_ood.shape)
	print(np.min(X_ood), np.max(X_ood))
	print(np.min(y_ood), np.max(y_ood))
	
	print("ID details: ", X_id.shape, y_id.shape)
	print(np.min(X_id), np.max(X_id))
	print(np.min(y_id), np.max(y_id))

	return X_id, y_id, X_ood, y_ood


class k_fold_generator:
	def __init__(self, y, fold, _seed_=11):
		self.nfold = fold
		self._seed_ = _seed_
		self.order = self.make_splittable(y)
		# print(self.order)
	
	def make_splittable(self, y):
		np.random.seed(self._seed_)
		sort_order = np.argsort(y)
		#permute the bins
		order = []
		
		n_bins = int(np.ceil(len(y)/self.nfold))
		for i in range(n_bins):
			if i == n_bins-1:
				start_index = i*self.nfold
				end_index = len(y)
			else:
				start_index = i*self.nfold
				end_index = (i+1)*self.nfold
			#index of the bin
			bin_indices = sort_order[start_index:end_index]
			bin_indices = np.random.permutation(bin_indices)
			
			order.append(bin_indices)
		
		order = np.concatenate(order)
		return order
	
	def get_kth_train_test_split(self, x, y, z, fno):
		#X is the list of all to be permuted
		train_indices = []
		test_indices = []
		
		#number of bins - for stratified splits
		n_bins = int(np.ceil(len(y)/self.nfold))
		
		for i in range(n_bins):
			if i == n_bins-1:
				start_index = i*self.nfold
				end_index = len(y)
			else:
				start_index = i*self.nfold
				end_index = (i+1)*self.nfold
			#index of the bin
			bin_indices = self.order[start_index:end_index]
			
			#if it is not the last bin
			if fno < len(bin_indices):
				test_indices.append(bin_indices[fno])
				bin_indices = np.delete(bin_indices, fno)
			
			train_indices.append(bin_indices)
		
		test_indices = np.array(test_indices)
		train_indices = np.concatenate(train_indices)
		print("Fold: {}(of {}), Test size: {}, Train size: {}, No. of train-test commoners: {}".format(fno+1, self.nfold, len(test_indices), len(train_indices), len(np.intersect1d(train_indices, test_indices))))
		
		x_train = x[train_indices]
		y_train = y[train_indices]
		z_train = z[train_indices]
		
		x_test = x[test_indices]
		y_test = y[test_indices]
		z_test = z[test_indices]
		
		return x_train, y_train, z_train, x_test, y_test, z_test
		
		
	def get_kth_train_test_split2(self, x, y, fno):
		#X is the list of all to be permuted
		train_indices = []
		test_indices = []
		
		#number of bins - for stratified splits
		n_bins = int(np.ceil(len(y)/self.nfold))
		
		for i in range(n_bins):
			if i == n_bins-1:
				start_index = i*self.nfold
				end_index = len(y)
			else:
				start_index = i*self.nfold
				end_index = (i+1)*self.nfold
			#index of the bin
			bin_indices = self.order[start_index:end_index]
			
			#if it is not the last bin
			if fno < len(bin_indices):
				test_indices.append(bin_indices[fno])
				bin_indices = np.delete(bin_indices, fno)
			
			train_indices.append(bin_indices)
		
		test_indices = np.squeeze(np.array(test_indices))
		#print(test_indices.shape)
		train_indices = np.concatenate(train_indices)
		#print(train_indices.shape)
		print("Fold: {}(of {}), Test size: {}, Train size: {}, No. of train-test commoners: {}".format(fno+1, self.nfold, len(test_indices), len(train_indices), len(np.intersect1d(train_indices, test_indices))))
		
		x_train = x[train_indices]
		y_train = y[train_indices]
		
		x_test = x[test_indices]
		y_test = y[test_indices]
		
		return x_train, y_train, x_test, y_test




if __name__=='__main__':
	# X_id, y_id, X_ood, y_ood, rescale_obj = load_id_ood_cam()
	read_cam("/data1/mainak/cancer_pred/data/cam16.npz", sub_samp=True)

	



