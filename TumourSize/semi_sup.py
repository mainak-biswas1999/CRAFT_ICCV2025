from plotter import *
from read_data import *
from resnet import *


def plot_distributions():
	_, y_id, _, y_ood = load_id_ood_cam(log_scale=False)
	print(y_id.shape, y_ood.shape, np.mean(y_id), np.std(y_id), np.mean(y_ood), np.std(y_ood))

	plot_logit_histograms(y_id, None, "./results/stats/label_dist_id_sep_new.png", title_='Label Distribution', xlabel_="tumour size", l1='ID (Cam16)', n_gmm=4)
	plot_logit_histograms(y_ood, None, "./results/stats/label_dist_ood_sep_new.png", title_='Label Distribution', xlabel_="tumour size", l1='OOD (Cam17)', n_gmm=4)
	# plt.hist(y_id, bins=10)
	# plt.savefig('./results/stats/label_hist_48.png')
	# plt.close()


def naive_model(check_train=True):
	n_epochs = 25
	n_fold = 4
	n_seeds = 1
	
	X_train, y_train, X_val, y_val, X_test, y_test = read_cam('./data/cam16.npz', True)
	
	fptr = open('./results/stats/naive_cam16_patched_subSamp.txt', 'w')
	pred_train = np.mean(y_train)*np.ones(y_train.shape)
	pred_val = np.mean(y_train)*np.ones(y_val.shape)
	pred_test = np.mean(y_train)*np.ones(y_test.shape)

	print("Train loss: {}".format(mse_loss(y_train, pred_train)), file=fptr)
	print("Val loss: {}".format(mse_loss(y_val, pred_val)), file=fptr)
	print("Test loss: {}".format(mse_loss(y_test, pred_test)), file=fptr)
	fptr.close()

	##########ood
	X_train, y_train, X_val, y_val, X_test, y_test = read_cam('./data/cam17.npz')
	fptr = open('./results/stats/naive_cam17_patched.txt', 'w')
	pred_train = np.mean(y_train)*np.ones(y_train.shape)
	pred_val = np.mean(y_train)*np.ones(y_val.shape)
	pred_test = np.mean(y_train)*np.ones(y_test.shape)

	print("Train loss: {}".format(mse_loss(y_train, pred_train)), file=fptr)
	print("Val loss: {}".format(mse_loss(y_val, pred_val)), file=fptr)
	print("Test loss: {}".format(mse_loss(y_test, pred_test)), file=fptr)
	fptr.close()


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


def return_labeled_subets_rnd(X, y, perc_label):
	n_lab = int(y.shape[0]*perc_label)
	indices = np.arange(y.shape[0])
	lab = np.random.choice(y.shape[0], n_lab, replace=False)
	ulab = np.delete(indices, lab)

	return X[lab], y[lab], X[ulab], y[ulab] 


def run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, m_saveloc, r_saveloc, pretrain_loc, seed=0, perc_label=0.5):
	np.random.seed(seed)
	X_train_ood, y_train_ood, _, _ = return_labeled_subets(X_train_ood, y_train_ood, perc_label)
	print(X_train_ood.shape)
	obj = resnet_model()
	obj.load_model(pretrain_loc)
	obj.train_model(X_train_ood, y_train_ood, m_saveloc, r_saveloc, num_epochs=10, val_data=X_val_ood, val_label=y_val_ood)

	del X_train_ood, y_train_ood
	del obj


def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, m_saveloc, r_saveloc):
	obj = resnet_model()
	obj.load_model(m_saveloc)

	y_train = y_train_ood
	y_val = y_val_ood
	y_test = y_test_ood
	
	pred_train_y = obj.predict(X_train_ood)
	pred_val_y = obj.predict(X_val_ood)
	pred_test_y = obj.predict(X_test_ood)

	plot_predictions_paper(pred_train_y[:, 0], y_train[:, 0], r_saveloc+"train_r_target_paper.png", title_='Train', addon="Coverage") #, )
	plot_predictions_paper(pred_val_y[:, 0], y_val[:, 0], r_saveloc+"val_r_target_paper.png", title_= 'Validation', addon="Coverage")
	plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Resnet: TL', addon="Tumour Coverage")
	del obj


if __name__=='__main__':
	import gc
	# naive_model()
	# plot_distributions()

	base_mloc = "./models/semi_sup/"
	base_rloc = "./results/semi_sup/"
	X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood = read_cam('/data1/mainak/cancer_pred/data/cam17.npz')

	for i in [1, 2, 5, 10]:
		if not os.path.exists(base_rloc+str(i)):
			os.mkdir(base_rloc+str(i))
		if not os.path.exists(base_mloc+str(i)):
			os.mkdir(base_mloc+str(i))
		for j in range(3):
			if not os.path.exists(base_rloc+str(i)+"/"+str(j)):
				os.mkdir(base_rloc+str(i)+"/"+str(j))
			if not os.path.exists(base_mloc+str(i)+"/"+str(j)):
				os.mkdir(base_mloc+str(i)+"/"+str(j))
			
			run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", pretrain_loc='./models/cam16_base_patched/', seed=j, perc_label=(0.01*i))
			run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
			
			gc.collect()
			tf.keras.backend.clear_session()