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

def save_npz_files_nwpu():
	X_train, y_train = read_data_nwpu('train')
	X_val, y_val = read_data_nwpu('val')
	X_test, _ = read_data_nwpu('test')

	np.savez('./data/nwpu-crowd/npz_files/data_nwpu.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test)

def save_npz_files_jhu():
	X_train, y_train = read_data_jhu('train')
	X_val, y_val = read_data_jhu('val')
	X_test, y_test = read_data_jhu('test')

	np.savez('./data/jhu_crowd_v2.0/npz_file/data_jhu.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

class Rescaling:
	def __init__(self, __type__='linear', a=0.0, b=1.0):
		self.type = __type__
		self.mu = None
		self.sigma = None
		
		self.a = a
		self.b = b
		self.vmax = None
		self.vmin = None
		
		self.vmin_1 = None
		self.vmax_1 = None
		
		self.vmin_2 = None
		self.vmax_2 = None
	
	# the target test distribution is upto 10_000 people and thus, training with it
	def set_params(self, eye_data):
		if self.type == 'linear':
			self.vmin = 0  #np.min(eye_data[:, 0])
			self.vmax = 10000   #np.max(eye_data[:, 0])
		elif self.type == 'linear2':
			self.vmin_1 = np.min(eye_data[:, 0])
			self.vmax_1 = np.max(eye_data[:, 0])
			self.vmin_2 = np.min(eye_data[:, 1])
			self.vmax_2 = np.max(eye_data[:, 1])
		elif self.type == 'z-score':
			self.mu = np.mean(eye_data)
			self.sigma = np.std(eye_data)
		else:
			print("{} as rescaling option doesn't exist!".format(type))
			sys.exit()
			
	def inv_scale(self, pred_data):
		if self.type == 'linear':
			inv_rescaled_edata = ((self.vmax - self.vmin)*pred_data - (self.a*self.vmax - self.b*self.vmin))/(self.b - self.a)
		elif self.type == 'linear2':
			inv_rescaled_edata = np.zeros(pred_data.shape)
			inv_rescaled_edata[:, 0] = ((self.vmax_1 - self.vmin_1)*pred_data[:, 0] - (self.a*self.vmax_1 - self.b*self.vmin_1))/(self.b - self.a)
			inv_rescaled_edata[:, 1] = ((self.vmax_2 - self.vmin_2)*pred_data[:, 1] - (self.a*self.vmax_2 - self.b*self.vmin_2))/(self.b - self.a)
		elif self.type == 'z-score':
			inv_rescaled_edata = self.mu + self.sigma*pred_data
		#else:
		#	sys.exit()
		return inv_rescaled_edata

	def rescale(self, need_to_rescale):
		if self.type == 'linear':
			rescaled_edata = ((self.b - self.a)*need_to_rescale + (self.a*self.vmax - self.b*self.vmin))/(self.vmax - self.vmin)
		elif self.type == 'linear2':
			rescaled_edata = np.zeros(need_to_rescale.shape)
			rescaled_edata[:, 0] = ((self.b - self.a)*need_to_rescale[:, 0] + (self.a*self.vmax_1 - self.b*self.vmin_1))/(self.vmax_1 - self.vmin_1)
			rescaled_edata[:, 1] = ((self.b - self.a)*need_to_rescale[:, 1] + (self.a*self.vmax_2 - self.b*self.vmin_2))/(self.vmax_2 - self.vmin_2)
		elif self.type == 'z-score':
			rescaled_edata = (need_to_rescale - self.mu)/self.sigma
		#else:
		#   print("{} as rescaling option doesn't exist!".format(type))
		#   sys.exit()
		return rescaled_edata

def read_data_jhu(split='train'):
	# options -- 'train'; 'val'; 'test'
	label_loc_detailed = "./data/jhu_crowd_v2.0/"+split+"/gt/"
	label_loc_img_level = "./data/jhu_crowd_v2.0/"+split+"/image_labels.txt"
	image_loc = "./data/jhu_crowd_v2.0/"+split+"/images/"

	images = []
	labels = []
	with open(label_loc_img_level, 'r') as txt_file:
		for (idx, line) in enumerate(txt_file):
			details = line.split(',')
			img = cv2.imread(image_loc + details[0] + ".jpg")
			# img = 1
			if img is not None:
				img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
				# print(img.shape)
				# cv2.imshow('JPG image', img)
				# cv2.waitKey(0)
				images.append([np.array(img)])
				labels.append(int(details[1]))
			# for faster reading
			# if idx == 100:
			# 	break
	
	images = (np.concatenate(images).astype('float32'))/255.0
	labels = np.expand_dims(np.array(labels), axis=-1)
	print(images.shape, labels.shape)
	return images, labels
	
def read_data_nwpu(split='train'):
	# options -- 'train'; 'val'; 'test'
	label_loc_detailed = "./data/nwpu-crowd/jsons/{}.json" 
	split_level = "./data/nwpu-crowd/"+split+".txt"
	image_loc = "./data/nwpu-crowd/images_part{}/{}.jpg"

	images = []
	labels = []
	with open(split_level, 'r') as txt_file:
		for (idx, line) in enumerate(txt_file):
			details = line.split(' ')
			# if split == 'train':
			img = cv2.imread(image_loc.format(np.min([int(1 + (int(details[0])-1)/1000), 5]), details[0]))
			# elif split == 'val':
			# 	img = cv2.imread(image_loc.format(4, details[0]))
			# else:
			# 	img = cv2.imread(image_loc.format(5, details[0]))

			# img = 1
			if img is not None:
				img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
				# print(img.shape)
				# cv2.imshow('JPG image', img)
				# cv2.waitKey(0)
				images.append([np.array(img)])
				if split != 'test':
					with open(label_loc_detailed.format(details[0])) as fptr:
						labels_json = json.load(fptr)
						labels.append(int(labels_json['human_num']))
			# for faster reading
			# if idx%500 == 0:
			# 	print(idx)
			# 	break
	
	images = (np.concatenate(images).astype('float32'))/255.0
	if split != 'test':
		labels = np.expand_dims(np.array(labels), axis=-1)

	# print(images.shape, labels.shape)
	return images, labels

def normalise_ys(y_train, y_val, y_test=None):
	rescale_obj = Rescaling()
	rescale_obj.set_params(y_train)
	# scale data
	y_train = rescale_obj.rescale(y_train)
	y_val = rescale_obj.rescale(y_val)
	if y_test is not None:
		y_test = rescale_obj.rescale(y_test)
	else:
		y_test = None
	# return the ys
	return y_train, y_val, y_test, rescale_obj


def stack_images(X, y=None, ids=None, N=6, max_y=10000):
	#send ids if not in order
	# ids = np.squeeze(ids)
	# labs = []
	X = X.reshape(int(X.shape[0]/N), N, X.shape[1], X.shape[2], X.shape[3])
	if y is not None:
		y_r = y.reshape((int(y.shape[0]/N), N, y.shape[1]))
		y_r = np.sum(y_r, axis=1)
		locs = np.where(np.squeeze(y_r) <= max_y)[0]
		y_r = y_r[locs]
		X = X[locs]
	else:
		y_r = []

	
	
	# dat = np.zeros((int(X.shape[0]/N), N, X.shape[1], X.shape[2], X.shape[3]))
	# for (n, uid) in enumerate(np.unique(ids)):
	# 	locs = np.where(ids==uid)[0]
	# 	dat[n] = X[locs]
	# 	labs.append(np.sum(y[locs]))

	# labs = np.expand_dims(np.array(labs), axis=-1)
	# print(np.where((X.reshape(int(X.shape[0]/N), N, X.shape[1], X.shape[2], X.shape[3]) == dat) == False))
	# print(np.where((np.sum(y_r, axis=1)==labs) == False))
	# print(y_r.shape, dat.shape)

	return X, y_r
	#return dat, labs

def get_jhu_data():
	data = np.load('/data6/mainak/people_counting_data/data/jhu_crowd_v2.0/npz_file/data_jhu.npz')
	
	X_train, y_train = stack_images(data['X_train'], data['y_train']) 
	X_val, y_val = stack_images(data['X_val'], data['y_val'])
	X_test, y_test = stack_images(data['X_test'], data['y_test'])
	print("OOD")
	print(np.min(X_train), np.max(X_train))
	
	# print(np.min(X_train), np.max(X_train))
	# y_train, y_val, y_test, rescale_obj = normalise_ys(y_train, y_val, y_test)

	return X_train, y_train, X_val, y_val, X_test, y_test   #, rescale_obj

def get_nwpu_data():
	# X_train, y_train = read_data_nwpu('train')
	# X_val, y_val = read_data_nwpu('val')
	# X_test, y_test = read_data_nwpu('test')
	data = np.load('/data6/mainak/people_counting_data/data/nwpu-crowd/npz_files/data_nwpu.npz')
	
	print("ID")
	X_train, y_train = stack_images(data['X_train'], data['y_train']) 
	X_val, y_val = stack_images(data['X_val'], data['y_val'])
	X_test, _ = stack_images(data['X_test'])

	print(np.min(X_train), np.max(X_train))
	
	# print(np.min(y_test), np.max(y_test))

	#, y_test.shape)
	# y_train, y_val, y_test, rescale_obj = normalise_ys(y_train, y_val)

	return X_train, y_train, X_val, y_val, X_test  # , rescale_obj

def load_id_ood_people(log_scale=False):
	# id -- nwpu
	X_train_id, y_train_id, X_val_id, y_val_id, X_test_id = get_nwpu_data()
	# ood -- jhu++
	X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood = get_jhu_data()

	if log_scale == True:
		y_train_ood = np.log(y_train_ood+1)
		y_val_ood = np.log(y_val_ood+1)
		y_test_ood = np.log(y_test_ood+1)
		
		y_train_id = np.log(y_train_id+1)
		y_val_id = np.log(y_val_id+1)


	print("OOD details: ", X_train_ood.shape, y_train_ood.shape, X_val_ood.shape, y_val_ood.shape, X_test_ood.shape, y_test_ood.shape)
	
	rescale_obj = Rescaling()
	rescale_obj.set_params(y_train_ood)
	
	y_train_ood = rescale_obj.rescale(y_train_ood)
	y_test_ood = rescale_obj.rescale(y_test_ood)
	y_val_ood = rescale_obj.rescale(y_val_ood)
	print(np.min(y_train_ood), np.max(y_train_ood))
	print(np.min(y_val_ood), np.max(y_val_ood))
	print(np.min(y_test_ood), np.max(y_test_ood))

	print("ID details: ", X_train_id.shape, y_train_id.shape, X_val_id.shape, y_val_id.shape, X_test_id.shape)
	y_train_id = rescale_obj.rescale(y_train_id)
	y_val_id = rescale_obj.rescale(y_val_id)
	print(np.min(y_train_id), np.max(y_train_id))
	print(np.min(y_val_id), np.max(y_val_id))

	return X_train_id.astype('float32')/255.0, y_train_id, X_val_id.astype('float32')/255.0, y_val_id, X_test_id.astype('float32')/255.0, X_train_ood.astype('float32')/255.0, y_train_ood, X_val_ood.astype('float32')/255.0, y_val_ood, X_test_ood.astype('float32')/255.0, y_test_ood, rescale_obj




if __name__=='__main__':
	# naive_model("./results/stats/naive_model.txt")
	X_train_id, y_train_id, X_val_id, y_val_id,  X_test_id, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people()
	# stack_images(X_val_id, y_val_id, ids_val_id)
	
	# save_npz_files_jhu()
	# X_train, y_train, X_val, y_val, X_test, y_test, rescale_obj = get_jhu_data()
	# save_npz_files_nwpu()
	# X_train, y_train, X_val, y_val, X_test, y_test, rescale_obj = get_nwpu_data()
	



