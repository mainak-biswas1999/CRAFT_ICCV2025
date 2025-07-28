import os
import numpy as np
import pandas as pd
import nilearn as nl
from nilearn.image import load_img
from nilearn.plotting import plot_anat
import sfcn_utils as dpu
from data_struct import *
from plotter import *
from sklearn.mixture import GaussianMixture
import pickle 

	
def store_fnames():
	datloc = './data/T1_RADC/'
	dat_dirs = os.listdir(datloc)
	ctr = 0
	t1_dat = []
	sub_ids = []
	for dname in dat_dirs:
		if os.path.exists(datloc+dname+'/t1_mni152.nii.gz'):	
			t1_dat.append(load_img(datloc+dname+'/t1_mni152.nii.gz'))
			sub_ids.append(dname)
			ctr += 1
	print(ctr)
	
	all_data = prep_t1_dat(t1_dat)
	age, cog_health = data_loader_mg_dmri_withcog("../DiffusionModels/datasets/mg_dmri_connectivity/", "../DiffusionModels/datasets/special_data/", sub_ids)
	all_data, age, _, sub_ids = clean_data_withcog(all_data, age, cog_health, sub_ids)
	
	details_radc_unlab = {
						  'sub_id': sub_ids,
						  'age': age,
							}
	with open('./data/radc_details', 'wb') as fname:
		pickle.dump(details_radc_unlab, fname)

	print(len(sub_ids), len(age))

def load_pkl():
	with open("./data/radc_details", 'rb') as fname:
		dct = pickle.load(fname)
	sub_ids = dct['sub_id']
	age = dct['age']
	index = np.where((age>=46.0) * (age<=86.0))[0]
	age = age[index]
	sub_ids = [sub_ids[i] for i in index]
	print(age.shape, len(sub_ids))
	details_radc_unlab = {
						  'sub_id': sub_ids,
						  'age': age,
							}
	with open('./data/radc_details_46_86', 'wb') as fname:
		pickle.dump(details_radc_unlab, fname)
	with open('./data/radc_subids.txt', 'w') as fname:
		for i in range(len(sub_ids)):
			print(sub_ids[i], file=fname)

def display_t1_img(t1_list):
	ind = np.random.randint(len(t1_list))
	plot_anat(t1_list[ind])

def check_data(t1_list):
	# data is perfect - (240, 256, 176) .* (1, 1, 2) mm
	all_shapes = [[], [], []]
	all_voxel_size = [[], [], []]
	for t1_file in t1_list:
		all_shapes[0].append(t1_file.dataobj.shape[0])
		all_shapes[1].append(t1_file.dataobj.shape[1])
		all_shapes[2].append(t1_file.dataobj.shape[2])

		dx, dy, dz = t1_file.header.get_zooms()
		all_voxel_size[0].append(dx)
		all_voxel_size[1].append(dy)
		all_voxel_size[2].append(dz)
	
	print(np.unique(all_voxel_size[0]), np.unique(all_voxel_size[1]), np.unique(all_voxel_size[2]))	
	print(np.unique(all_shapes[0]), np.unique(all_shapes[1]), np.unique(all_shapes[2]))

def min_cross_val_error(data, fname):
	np.random.seed(25)
	test_list = np.random.choice(data.shape[0], int(0.25*data.shape[0]), replace=False)
	fit_cv = data[test_list]
	fit_train = np.delete(data, test_list, axis=0)
	print(fit_train.shape, fit_cv.shape)
	log_likelihood_train = []
	log_likelihood_cv = []
	for i in range(1, 15):
		p_y_gmm = GaussianMixture(n_components=i, random_state=0).fit(fit_train)
		log_likelihood_train.append(p_y_gmm.score(fit_train))
		log_likelihood_cv.append(p_y_gmm.score(fit_cv))

	plt.plot(np.arange(1, 15), log_likelihood_train, label='train')
	plt.plot(np.arange(1, 15), log_likelihood_cv, label='cv')
	plt.xlabel("Number of GM components")
	plt.ylabel("log-likelihood")
	plt.legend(loc='lower left')
	plt.savefig(fname)
	plt.close()

def prep_t1_dat(t1_list):
	# data is perfect - (240, 256, 176) .* (1, 1, 2) mm
	all_data = []
	for t1_file in t1_list:
		data = np.array(t1_file.dataobj)

		data = data/data.mean()
		# print(data.shape)
		data = dpu.crop_center(data, (160, 192, 160))
		all_data.append([data])
	all_data = np.concatenate(all_data)
	# print(np.max(all_data), np.min(all_data), np.mean(all_data), np.std(all_data))
	return all_data

def data_loader_mg_dmri_withcog(dataloc, health_dataloc, sub_ids):
	#this will load the labels of the dataset
	labels = pd.read_csv(dataloc+"dmri_data_labels.csv")
	labels_health = pd.read_excel(health_dataloc+"dataset_611_long_04-23-2019.xlsx")
	
	
	sub_wise_labels_health = {
							  'sub_id': None,
							  'cog_health': None
							 }
	
	sub_wise_labels_health['cog_health'] = labels_health['dcfdx'].tolist()
	proj_id = labels_health['projid'].tolist()
	fu_year = labels_health['fu_year'].tolist()
	
	sub_wise_labels_health['sub_id'] = ['{}_{}'.format(str(proj_id[i]).rjust(8, '0'), str(fu_year[i]).rjust(2, '0')) for i in range(len(fu_year))]
	
	#print(sub_wise_labels_health['sub_id'][0:10])
	sub_wise_labels = {
						'sub_id': None,
						'age': None
					}
	
	sub_wise_labels['age'] = labels['age_at_visit'].tolist()
	sub_wise_labels['sub_id'] = labels['sub_id'].tolist()
	
	
	age = []
	cog_health = []
	ctr = 0
	ctr2 = 0
	#list the subdirectories
	list_dir = os.listdir(dataloc)
	for folder_name in sub_ids:
		#age retrieval
		if folder_name in sub_wise_labels['sub_id']:
			age_index = sub_wise_labels['sub_id'].index(folder_name)
			age.append(sub_wise_labels['age'][age_index])
		else:
			age.append(-1)
		
		#cogscore distribution
		if folder_name in sub_wise_labels_health['sub_id']:
			cogscore_index = sub_wise_labels_health['sub_id'].index(folder_name)
			if np.isnan(sub_wise_labels_health['cog_health'][cogscore_index]):
				cog_health.append(-1)
			else:
				cog_health.append(sub_wise_labels_health['cog_health'][cogscore_index])
			
		else:
			cog_health.append(-1)
	
	cog_health = np.array(cog_health, dtype='int32')
	age = np.array(age)  
	#print(cog_health.shape, conn_matrices.shape, age.shape)
	#print(np.unique(cog_health))
	
	return age, cog_health

def clean_data_withcog(mat, age, cog_health, sub_ids):
	#indices with -1.
	no_info = np.where(np.logical_or((age == -1), (cog_health == -1)))[0]
	
	info_index = np.delete(np.arange(age.shape[0]), no_info)
	
	#age without -1
	age_trimmed = age[info_index]
	mat_trimmed = mat[info_index]
	cog_health_trimmed = cog_health[info_index]
	sub_ids = [sub_ids[i] for i in info_index]
	return mat_trimmed, age_trimmed, cog_health_trimmed, sub_ids

def load_radc():
	datloc = './data/T1_RADC/'
	dat_dirs = os.listdir(datloc)
	ctr = 0
	t1_dat = []
	sub_ids = []
	for dname in dat_dirs:
		if os.path.exists(datloc+dname+'/t1_brain_restore.nii.gz'):	
			t1_dat.append(load_img(datloc+dname+'/t1_brain_restore.nii.gz'))
			sub_ids.append(dname)
			ctr += 1
	print(ctr)
	
	all_data = prep_t1_dat(t1_dat)
	age, cog_health = data_loader_mg_dmri_withcog("../DiffusionModels/datasets/mg_dmri_connectivity/", "../DiffusionModels/datasets/special_data/", sub_ids)
	all_data, age, _, _ = clean_data_withcog(all_data, age, cog_health, sub_ids)
	age = np.expand_dims(age, axis=-1)
	print(all_data.shape, age.shape)
	return all_data, age

def load_data():
	datloc = './data/T1_TLSA/'
	dat_dirs = os.listdir(datloc)
	ctr = 0
	t1_dat = []
	sub_ids = []
	for dname in dat_dirs:
		if os.path.exists(datloc+dname+'/t1_brain_restore.nii.gz'):
			t1_dat.append(load_img(datloc+dname+'/t1_brain_restore.nii.gz'))
			sub_ids.append(int(dname[0:3]))
			ctr += 1
	print(ctr)
	# check_data(t1_dat)
	# data is as expected
	# display_t1_img(t1_dat)
	all_data = prep_t1_dat(t1_dat)
	return all_data, sub_ids

def loader_multimodal_tlsa(dataloc):
	data_loader = TLSA_data()
	conn_mats, age, subids = data_loader.data_loader_conn(dataloc,"./Results/iclr/meta/", "./Results/iclr/meta/", ret_subid=True)
	t1_data, sub_ids = load_data()
	
	age_t1 = []
	conn_data = []
	subid_data = []
	for curr_id in sub_ids:
		subid_data.append([curr_id])
		age_t1.append(age[np.where(subids == curr_id)[0]])
		conn_data.append(conn_mats[np.where(subids == curr_id)[0]])

	age_t1 = np.array(age_t1)
	conn_data = np.concatenate(conn_data)
	subid_data = np.array(subid_data)

	print(subid_data.shape, t1_data.shape, conn_data.shape, age_t1.shape)
	print(np.min(age_t1), np.max(age_t1))
	
	return subid_data, t1_data, conn_data, age_t1

def loader_t1_tlsa(dataloc, plot_age_dist=False):
	data_loader = TLSA_data()
	_, age, subids = data_loader.data_loader_conn(dataloc,"./Results/iclr/meta/", "./Results/iclr/meta/", ret_subid=True)
	t1_data, sub_ids = load_data()
	age_t1 = []
	for curr_id in sub_ids:
		#print(curr_id, age[np.where(subids == curr_id)[0]], subids[np.where(subids == curr_id)[0]])
		age_t1.append(age[np.where(subids == curr_id)[0]])
	
	age_t1 = np.array(age_t1)
	if plot_age_dist:
		min_cross_val_error(age_t1, "./Results/iclr/metaa/crossval_error.png")
		p_y_gmm = GaussianMixture(n_components=3, random_state=0).fit(age_t1)
		test_fit_y = np.expand_dims(np.linspace(np.min(age_t1)-2, np.max(age_t1)+2, 1000), axis=-1)
		test_fit_py = np.exp(p_y_gmm.score_samples(test_fit_y))
		plot_histo(age_t1, test_fit_y, test_fit_py, 40, "age (in years)", "Density", "age", "./Results/iclr/meta/age_distr.png")
	print(t1_data.shape, age_t1.shape)
	print(np.min(age_t1), np.max(age_t1))
	
	return t1_data, age_t1

def load_2_datasets():
	tlsa_t1, age_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	#print(np.min(age_tlsa), np.max(age_tlsa))
	radc_t1, age_radc = load_radc()
	index = np.where((age_radc>=np.min(age_tlsa)) * (age_radc<=np.max(age_tlsa)))[0]
	radc_t1 = radc_t1[index]
	age_radc = age_radc[index]
	print(radc_t1.shape, age_radc.shape)
	return tlsa_t1, age_tlsa, radc_t1, age_radc

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

# loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2") #, True)
# load_radc()
# load_2_datasets()

if __name__=="__main__":
	# store_fnames()
	load_pkl()