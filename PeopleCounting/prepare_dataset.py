from read_data import *

def make_sub_images(Img_data, tot_humans, Gt_data):
	#adapted from TransCrowd dataset
	img_list = []
	label_list = []
	if Img_data.shape[1] >= Img_data.shape[0]:
		rate_1 = 1152.0 / Img_data.shape[1]
		rate_2 = 768 / Img_data.shape[0]
		Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
		if tot_humans > 0:
			Gt_data[:, 0] = Gt_data[:, 0] * rate_1
			Gt_data[:, 1] = Gt_data[:, 1] * rate_2

	elif Img_data.shape[0] > Img_data.shape[1]:
		rate_1 = 1152.0 / Img_data.shape[0]
		rate_2 = 768.0 / Img_data.shape[1]
		Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
		if tot_humans > 0:
			Gt_data[:, 0] = Gt_data[:, 0] * rate_2
			Gt_data[:, 1] = Gt_data[:, 1] * rate_1
		# print(img_path)


	kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
	# print(Gt_data)
	if tot_humans > 0:	
		for count in range(0, len(Gt_data)):
			if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
				kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

			# kpoint[np.min([int(Gt_data[count][1]), Img_data.shape[0]-1]), np.min([int(Gt_data[count][0]), Img_data.shape[1]-1])] = 1

	height, width = Img_data.shape[0], Img_data.shape[1]

	m = int(width / 384)
	n = int(height / 384)
	
	for i in range(0, m):
		for j in range(0, n):
			crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
			crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
			gt_count = np.sum(crop_kpoint)

			img_list.append([crop_img])
			label_list.append(gt_count)

	return img_list, label_list


def prep_nwpu(split='train'):
	# options -- 'train'; 'val'; 'test'
	label_loc_detailed = "./data/nwpu-crowd/jsons/{}.json" 
	split_level = "./data/nwpu-crowd/"+split+".txt"
	image_loc = "./data/nwpu-crowd/images_part{}/{}.jpg"

	images = []
	labels = []
	img_ids = []
	with open(split_level, 'r') as txt_file:
		for (idx, line) in enumerate(txt_file):
			details = line.split(' ')
			img = cv2.imread(image_loc.format(np.min([int(1 + (int(details[0])-1)/1000), 5]), details[0]))
			if img is not None:
				if split != 'test':
					with open(label_loc_detailed.format(details[0])) as fptr:
						labels_json = json.load(fptr)
						tot_humans = labels_json['human_num']
						if len(labels_json['points']) != 0:
							pos_humans = np.vstack(labels_json['points'])
						else:
							pos_humans = []

						# print(labels_json['points'])
						sub_images, n_humans = make_sub_images(img, tot_humans, pos_humans)
						if tot_humans != int(np.sum(n_humans)):
							print("Number of humans in the sub-images and the super-image don't add up for: {}, diff={}/{}".format(details[0], tot_humans - np.sum(n_humans), tot_humans))

						img_id = list(np.ones(len(sub_images))*idx)
						# print(tot_humans, np.sum(n_humans), sub_images[0].shape, len(sub_images), n_humans)
						
				
				else:
					# labels aren't available
					sub_images, n_humans = make_sub_images(img, 0, [])
					img_id = list(np.ones(len(sub_images))*idx)

				images += sub_images
				labels += n_humans
				img_ids += img_id

			# if idx%500 == 0:
			# 	print(idx)
			# if idx == 10:
			# 	break
	
	# print(len(images))
	images = np.vstack(images).astype('uint8')   # .astype('float32'))/255.0
	img_ids = np.expand_dims(np.array(img_ids), axis=-1)
	labels = np.expand_dims(np.array(labels), axis=-1)

	print(images.shape, np.min(images), np.max(images), labels.shape, img_ids.shape, np.unique(img_ids).shape[0])
	return images, labels, img_ids


def save_npz_files_nwpu_subImg():
	X_train, y_train, ids_train = prep_nwpu('train')
	X_val, y_val, ids_val = prep_nwpu('val')
	X_test, _, ids_test = prep_nwpu('test')

	np.savez('./data/nwpu-crowd/npz_files/data_nwpu.npz', X_train=X_train, y_train=y_train, ids_train=ids_train, X_val=X_val, y_val=y_val, ids_val=ids_val, X_test=X_test, ids_test=ids_test)


def make_gt_jhu(floc):
	annPoints = []
	with open(floc) as txt_ptr:
		for (idx, line) in enumerate(txt_ptr):
			details = line.split(' ')
			annPoints.append([int(details[0]), int(details[1])])
	if len(annPoints) > 0:
		return np.vstack(annPoints)
	else:
		return annPoints

def prep_jhu(split='train'):
	# options -- 'train'; 'val'; 'test'
	label_loc_detailed = "./data/jhu_crowd_v2.0/"+split+"/gt/"
	label_loc_img_level = "./data/jhu_crowd_v2.0/"+split+"/image_labels.txt"
	image_loc = "./data/jhu_crowd_v2.0/"+split+"/images/"

	images = []
	labels = []
	img_ids = []
	with open(label_loc_img_level, 'r') as txt_file:
		for (idx, line) in enumerate(txt_file):
			details = line.split(',')
			img = cv2.imread(image_loc + details[0] + ".jpg")
			# img = 1
			if img is not None:
				tot_humans = int(details[1])
				pos_humans = make_gt_jhu(label_loc_detailed+details[0]+".txt")
				# print(pos_humans)
				# print(labels_json['points'])
				sub_images, n_humans = make_sub_images(img, tot_humans, pos_humans)
				if tot_humans != int(np.sum(n_humans)):
					print("Number of humans in the sub-images and the super-image don't add up for: {}, diff={}/{}".format(details[0], tot_humans - np.sum(n_humans), tot_humans))

				img_id = list(np.ones(len(sub_images))*idx)
				# print(tot_humans, np.sum(n_humans), sub_images[0].shape, len(sub_images), n_humans)

				images += sub_images
				labels += n_humans
				img_ids += img_id
			# for faster reading
			# if idx == 10:
			# 	break
	
	images = np.vstack(images).astype('uint8')   # .astype('float32'))/255.0
	img_ids = np.expand_dims(np.array(img_ids), axis=-1)
	labels = np.expand_dims(np.array(labels), axis=-1)

	print(images.shape, np.min(images), np.max(images), labels.shape, img_ids.shape, np.unique(img_ids).shape[0])
	return images, labels, img_ids


def save_npz_files_jhu_subImg():
	X_train, y_train, ids_train = prep_jhu('train')
	X_val, y_val, ids_val = prep_jhu('val')
	X_test, y_test, ids_test = prep_jhu('test')

	np.savez('./data/jhu_crowd_v2.0/npz_file/data_jhu.npz', X_train=X_train, y_train=y_train, ids_train=ids_train, X_val=X_val, y_val=y_val, ids_val=ids_val, X_test=X_test, y_test=y_test, ids_test=ids_test)


if __name__=='__main__':
	# save_npz_files_nwpu_subImg()
	save_npz_files_jhu_subImg()