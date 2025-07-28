from sfcn_tensorflow import *
from loader import *
from data_struct import *
from plotter import *
from plot_paper import *
import matplotlib
import sys

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


def call_k_fold_tlsa(base_rloc_r, base_mloc_r, perc_label=0.2, check_train=False):
	n_epochs = 50
	n_fold = 4
	age_range_loc = "./Results/iclr/meta/tlsa_age_scale.npy"
	if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
		os.mkdir(base_rloc_r+str(int(perc_label*100)))
	if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
		os.mkdir(base_mloc_r+str(int(perc_label*100)))
	base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
	base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"

	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
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
			# print(wrongly_sample_index.shape)
			np.random.seed(j)
			X_train_foldi_sub_samp, y_train_foldi_sub_samp, _, _ = return_labeled_subets(X_train_foldi, y_train_foldi, perc_label)

			
			print(X_train_foldi_sub_samp.shape, y_train_foldi_sub_samp.shape)
			
			plot_histo(y_train_foldi_sub_samp.ravel(), saveloc=base_rloc+"fold_"+str(i+1)+"/histo_fake.png")
			plot_histo(y_train_foldi.ravel(), saveloc=base_rloc+"fold_"+str(i+1)+"/histo_real.png")
			# return
			# continue
			# print(y_train_foldi.shape, y_test_foldi.shape)	
			# return

			tlsa_act.append(y_test_foldi)
			obj_model = SFCN_tf(pretrained=True, to_train_full=True)
			if check_train == True and os.path.exists(base_mloc+"fold_"+str(i+1)):
				obj_model.load_model(base_mloc+"fold_"+str(i+1)+"/")

			else:
				obj_model.add_mse_head()
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


def call_k_fold_naivemodel(base_rloc_r, base_mloc_r, perc_label=0.2):
	n_fold = 4
	if not os.path.exists(base_rloc_r+str(int(perc_label*100))):
		os.mkdir(base_rloc_r+str(int(perc_label*100)))
	if not os.path.exists(base_mloc_r+str(int(perc_label*100))):
		os.mkdir(base_mloc_r+str(int(perc_label*100)))
	base_mloc_r = base_mloc_r+str(int(perc_label*100))+"/"
	base_rloc_r = base_rloc_r+str(int(perc_label*100))+"/"

	X_train_tlsa, y_train_tlsa = loader_t1_tlsa("/hdd_home/mainak/Brain_AGE/data/sub_details_new_pruned_2")
	X_train_tlsa = np.expand_dims(X_train_tlsa, axis=-1)
	
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
			# print(wrongly_sample_index.shape)
			np.random.seed(j)
			X_train_foldi_sub_samp, y_train_foldi_sub_samp, _, _ = return_labeled_subets(X_train_foldi, y_train_foldi, perc_label)

			
			print("-----------------fold i------------------", file=pdetails)
			pred_y_train_foldi = np.mean(y_train_foldi_sub_samp)

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


if __name__ == '__main__':
	perc_labels = [0.2, 0.4, 0.6]
	for perc_label in perc_labels:
		# call_k_fold_tlsa("./Results/rebuttal/perc_sup/", "./Models/rebuttal/perc_sup/", perc_label=perc_label, check_train=False)
		call_k_fold_naivemodel("./Results/rebuttal/naive/", "./Models/rebuttal/naive/", perc_label=perc_label)
