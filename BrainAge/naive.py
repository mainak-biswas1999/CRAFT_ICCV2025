from sfcn_tensorflow import *
from loader import *
from data_struct import *
from plotter import *
from plot_paper import *
import matplotlib

matplotlib.use('Agg')

def plot_histo(density_est_x, xlabel_="Age (in years)", y_label="Density", saveloc="./histo.png"):
		#plot the histogram of velocities
	
	plt.hist(density_est_x, bins=40, density=True)
	

	plt.xlabel(xlabel_, fontsize=16)
	plt.ylabel(y_label, fontsize=16)
	plt.savefig(saveloc, dpi=200, bbox_inches="tight")
	plt.close()

def get_norm(x, y, l=1):
	x = np.squeeze(x)
	y = np.squeeze(y)
	return (np.mean(np.abs(x-y)**l))**(1./l)

def test_sfcn(use_pretrained=True, saveloc="./Results/naive/pred_sfcn_tlsa_correctedPrep.png"):
	sfcn = SFCN_tf(pretrained=use_pretrained)
	tlsa_t1, age_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	# tlsa_t1, age_tlsa, radc_t1, age_radc = load_2_datasets()
	# print(t1_data.shape, np.min(age_t1), np.max(age_t1))

	tlsa_t1 = np.expand_dims(tlsa_t1, axis=-1)
	# radc_t1 = np.expand_dims(radc_t1, axis=-1)
	pred_age_tlsa = sfcn.predict_orig(tlsa_t1)
	# pred_age_radc = sfcn.predict_orig(radc_t1)

	a_bins = np.expand_dims(np.arange(46.5, 86.5), axis=-1)
	# plot_predictions(pred_age @ a_bins, age_t1, saveloc, title_="Predictions: SFCN", addon="Age (in years)")
	# plot_predictions_paper(pred_age_radc @ a_bins, age_radc, saveloc+"radc.png", title_="Directly applied to RADC", addon="Age (in years)")
	plot_predictions_paper(pred_age_tlsa @ a_bins, age_tlsa, saveloc+"tlsa.png", title_="SFCN (UKBio)", addon="Age (in years)")
	
def call_k_fold_tlsa(check_train=False):
	n_epochs = 50
	n_fold = 4
	frac = 0.33
	base_rloc_r = "./Results/iclr/finetune_subsample_0.33/"
	base_mloc_r = "./Models/iclr/finetune_subsample_0.33/"
	age_range_loc = "./Results/iclr/meta/tlsa_age_scale.npy"
	
	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)
	
	#print(y_train_tlsa)
	#return
	for j in range(5):
		obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
		#return
		base_rloc = base_rloc_r + str(j) + "/"
		base_mloc = base_mloc_r + str(j) + "/"
		if not os.path.exists(base_rloc):
			os.mkdir(base_rloc)
		if not os.path.exists(base_mloc):
			os.mkdir(base_mloc)

		pdetails = open(base_rloc+"self_kfold_tlsa.txt", 'w')
		tlsa_act = []
		tlsa_pred = []
		for i in range(n_fold):
			#make places to save
			print("-------------------------------------Fold {}--------------------------------------------".format(i+1), file=pdetails)
			if not os.path.exists(base_rloc+"fold_"+str(i+1)):
				os.mkdir(base_rloc+"fold_"+str(i+1))
			if not os.path.exists(base_mloc+"fold_"+str(i+1)):
				os.mkdir(base_mloc+"fold_"+str(i+1))
			#train the model
			X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
			
			wrongly_sample_index = np.where(y_train_foldi>0.5)[0]
			# print(wrongly_sample_index.shape)
			np.random.seed(j)
			subselect_data = wrongly_sample_index[np.random.choice(len(wrongly_sample_index), int(frac*len(wrongly_sample_index)), replace=False)]
			selected_index = np.concatenate([np.where(y_train_foldi<=0.5)[0], subselect_data])

			X_train_foldi_sub_samp = X_train_foldi[selected_index]
			y_train_foldi_sub_samp = y_train_foldi[selected_index]
			
			print(X_train_foldi_sub_samp.shape, y_train_foldi_sub_samp.shape)
			plot_histo(y_train_foldi_sub_samp.ravel(), saveloc=base_rloc+"fold_"+str(i+1)+"/histo_fake.png")
			plot_histo(y_train_foldi.ravel(), saveloc=base_rloc+"fold_"+str(i+1)+"/histo_real.png")
			# continue
			# print(y_train_foldi.shape, y_test_foldi.shape)	
			# return

			tlsa_act.append(y_test_foldi)
			obj_model = SFCN_tf(pretrained=True, to_train_full=True)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				obj_model.add_mse_head()
				# obj_model.train_model(X_train_foldi, y_train_foldi, base_mloc+"fold_"+str(i+1)+"/", base_rloc+"fold_"+str(i+1)+"/", num_epochs=n_epochs)
				obj_model.train_model(X_train_foldi_sub_samp, y_train_foldi_sub_samp, base_mloc+"fold_"+str(i+1)+"/", base_rloc+"fold_"+str(i+1)+"/", num_epochs=n_epochs) 
			
			#plot training curves
			pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
			pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
			del obj_model
			
			
			print(pred_y_train_foldi.shape, pred_y_test_foldi.shape)
			tlsa_pred.append(pred_y_test_foldi)
			
			plot_predictions(pred_y_train_foldi, inv_prep(y_train_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/train_tlsa.png", title_="Train: SFCN finetune", addon= "Age (in years)")
			plot_predictions(pred_y_test_foldi, inv_prep(y_test_foldi, age_range_loc), base_rloc+"fold_"+str(i+1)+"/test_tlsa.png", title_="Test: SFCN finetune", addon= "Age (in years)")
		
		tlsa_act = np.concatenate(tlsa_act)
		tlsa_pred = np.concatenate(tlsa_pred)

		# plot_predictions(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results.png", title_="Test: SFCN finetune", addon="Age (in years)")
		plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper_Square.png", title_="Finetuning", addon="Age")
		pdetails.close()

def call_k_fold_naivemodel(check_train=False):
	n_fold = 4
	frac = 0.10
	base_rloc = "./Results/iclr/naive/naive_pred_subsample2.txt"
	
	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	# y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

	#print(y_train_tlsa)
	#return
	
	obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
	# pdetails = open(base_rloc+"naive_pred_tlsa.txt", 'w')
	pdetails = open(base_rloc, 'w')
	#return 
	tlsa_act = []
	tlsa_pred = []
	for i in range(n_fold):
		#make places to save
		X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
		
		wrongly_sample_index = np.where(y_train_foldi>66)[0]
		# print(wrongly_sample_index.shape)
		np.random.seed(0)
		subselect_data = wrongly_sample_index[np.random.choice(len(wrongly_sample_index), int(frac*len(wrongly_sample_index)), replace=False)]
		selected_index = np.concatenate([np.where(y_train_foldi<=66)[0], subselect_data])

		y_train_foldi = y_train_foldi[selected_index]

		print(np.min(y_test_foldi), np.max(y_train_foldi))
		print("-----------------fold i------------------", file=pdetails)
		pred_y_train_foldi = np.mean(y_train_foldi)

		print("Train MAE = {}, MSE = {} - \nTest MAE = {}, MSE = {}".format(get_norm(y_train_foldi, pred_y_train_foldi), get_norm(y_train_foldi, pred_y_train_foldi, 2), get_norm(y_test_foldi, pred_y_train_foldi), get_norm(y_test_foldi, pred_y_train_foldi, 2)), file=pdetails)
		tlsa_pred.append(pred_y_train_foldi*np.ones(y_test_foldi.shape))
		tlsa_act.append(y_test_foldi)
		
	tlsa_act = np.concatenate(tlsa_act)
	tlsa_pred = np.concatenate(tlsa_pred)
	print("\nOverall Test MAE = {}, MSE = {}".format(get_norm(tlsa_act, tlsa_pred), get_norm(tlsa_act, tlsa_pred, 2)), file=pdetails)
	pdetails.close()



	pdetails.close()

def call_k_fold_tlsa_paper(check_train=True):
	n_epochs = 50
	n_fold = 4
	n_seeds = 5
	base_rloc_r = "./Results/iclr/finetune/"
	base_mloc_r = "./Models/iclr/finetune/"
	age_range_loc = "./Results/iclr/meta/tlsa_age_scale.npy"
	
	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

	#print(y_train_tlsa)
	#return
	for j in range(5):
		obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
		#return
		base_rloc = base_rloc_r + str(j) + "/"
		base_mloc = base_mloc_r + str(j) + "/"
		if not os.path.exists(base_rloc):
			os.mkdir(base_rloc)
		if not os.path.exists(base_mloc):
			os.mkdir(base_mloc)
		tlsa_act = []
		tlsa_pred = []
		for i in range(n_fold):
			#make places to save
			if not os.path.exists(base_rloc+"fold_"+str(i+1)):
				os.mkdir(base_rloc+"fold_"+str(i+1))
			if not os.path.exists(base_mloc+"fold_"+str(i+1)):
				os.mkdir(base_mloc+"fold_"+str(i+1))
			#train the model
			X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
			
			# print(y_train_foldi.shape, y_test_foldi.shape)	
			# return
			tlsa_act.append(y_test_foldi)
			np.random.seed(j)
			obj_model = SFCN_tf(pretrained=False, to_train_full=True)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")
			else:
				obj_model.add_mse_head()
				obj_model.train_model(X_train_foldi, y_train_foldi, base_mloc+"fold_"+str(i+1)+"/", base_rloc+"fold_"+str(i+1)+"/", num_epochs=n_epochs) 
			
			#plot training curves
			pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
			pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
			del obj_model
			
			tlsa_pred.append(pred_y_test_foldi)
			
		tlsa_act = np.concatenate(tlsa_act)
		tlsa_pred = np.concatenate(tlsa_pred)
		plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper2.png", title_="N=188 (SFCN - TLSA)", addon="Age")

def call_k_fold_tlsa_paper_radc(check_train=True):
	n_epochs = 20
	n_fold = 5
	n_seeds = 5
	base_rloc_r = "./Results/pilots/radc_scratch/"
	base_mloc_r = "./Models/pilots/radc_scratch/"
	age_range_loc = "./Results/pilots/radc/tlsa_age_scale.npy"
	
	X_train_tlsa, y_train_tlsa = load_radc()
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	#print(y_train_tlsa)
	y_train_tlsa = normalize_single_age(y_train_tlsa, age_range_loc)

	#print(y_train_tlsa)
	#return
	for j in range(1):
		obj_data = k_fold_generator(np.squeeze(y_train_tlsa), n_fold)
		#return
		base_rloc = base_rloc_r + str(j) + "/"
		base_mloc = base_mloc_r + str(j) + "/"
		if not os.path.exists(base_rloc):
			os.mkdir(base_rloc)
		if not os.path.exists(base_mloc):
			os.mkdir(base_mloc)
		tlsa_act = []
		tlsa_pred = []
		for i in range(n_fold):
			#make places to save
			if not os.path.exists(base_rloc+"fold_"+str(i+1)):
				os.mkdir(base_rloc+"fold_"+str(i+1))
			if not os.path.exists(base_mloc+"fold_"+str(i+1)):
				os.mkdir(base_mloc+"fold_"+str(i+1))
			#train the model
			X_train_foldi, y_train_foldi, X_test_foldi, y_test_foldi = obj_data.get_kth_train_test_split2(X_train_tlsa, y_train_tlsa, i)
			
			# print(y_train_foldi.shape, y_test_foldi.shape)	
			# return

			tlsa_act.append(y_test_foldi)
			np.random.seed(j)
			obj_model = SFCN_tf(pretrained=False, to_train_full=True)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				obj_model.add_mse_head()
				obj_model.train_model(X_train_foldi, y_train_foldi, base_mloc+"fold_"+str(i+1)+"/", base_rloc+"fold_"+str(i+1)+"/", num_epochs=n_epochs) 
			
			#plot training curves
			pred_y_train_foldi = inv_prep(obj_model.predict_mse(X_train_foldi), age_range_loc)
			pred_y_test_foldi = inv_prep(obj_model.predict_mse(X_test_foldi), age_range_loc)
			del obj_model
			
			
			tlsa_pred.append(pred_y_test_foldi)
			
		tlsa_act = np.concatenate(tlsa_act)
		tlsa_pred = np.concatenate(tlsa_pred)

		plot_predictions_paper(tlsa_pred, inv_prep(tlsa_act, age_range_loc), base_rloc+"overall_results_paper.png", title_="RADC: SFCN", addon="Age (in years)")


def save_multimodal_features(use_pretrained=True, saveloc="./Results/naive/pred_sfcn_tlsa_correctedPrep.png"):
	bs = 8
	sfcn = SFCN_tf(pretrained=use_pretrained)
	sfcn.load_model("./Models/iclr/cdist_0.1_addradc3/3/fold_1/Decoder_cdist/")
	sfcn_feat_extractor = sfcn.get_multimodal_feat_extractor()
	sfcn_feat_extractor.summary()
	subid_data, t1_data, conn_data, age_t1 = loader_multimodal_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	t1_data = np.expand_dims(t1_data, axis=-1)
	itermax = int(np.ceil(subid_data.shape[0]/bs))
	t1_feat_data = []
	for i in range(itermax):
		st = i*bs
		if  i==itermax-1:
			en = subid_data.shape[0]
		else:
			en = (i+1)*bs
		t1_feat_data.append(sfcn_feat_extractor(t1_data[st:en]).numpy())

	t1_feat_data = np.concatenate(t1_feat_data)
	print(subid_data.shape, t1_feat_data.shape, conn_data.shape, age_t1.shape)
	np.savez('./sfcn_latents/multimodal_tlsa_best.npz', subid=subid_data, t1_feat=t1_feat_data, conn=conn_data, age=age_t1)

def load_multimodal_features():
	tlsa = np.load("./data/multimodal_tlsa.npz")
	subid = tlsa['subid']
	t1_feat = tlsa['t1_feat']
	conn = tlsa['conn']
	age = tlsa['age']
	print(subid.shape, t1_feat.shape, conn.shape, age.shape)


if __name__ == '__main__':
	# load_multimodal_features()
	# save_multimodal_features()
	# call_k_fold_tlsa_paper_radc(False)
	# call_k_fold_naivemodel(check_train=False)
	call_k_fold_tlsa(True)
	# call_k_fold_tlsa_paper(True)
	# test_sfcn(use_pretrained=True, saveloc="./Results/naive/pred_sfcn_tlsa_correctedPrep.png")
	# test_sfcn(True, "./Results/iclr/stats/new_dat_paper_")
	# test_sfcn(False, "./Results/naive/pred_random_sfcn_tlsa_correctedPrep.png")
	# call_k_fold_tlsa(check_train=False)
	# call_k_fold_naivemodel(check_train=True)
