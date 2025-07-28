import numpy as np
import matplotlib.pyplot as plt
import pingouin as png
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm
def plot_predictions_paper(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
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

	
	plt.xlabel("Actual "+addon, fontsize=45, fontname="Myriad Pro")
	plt.ylabel("Predicted "+addon, fontsize=45, fontname="Myriad Pro")
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
	density = np.log(-1*p_y_gmm.score_samples(mesh).reshape(mX.shape))
	
	
	alpha = 1
	density2 = (-1*p_y_gmm.score_samples(data))   #**0.000005
	#density2 = ((density2 - np.min(density2))/(np.max(density2) - np.min(density2)))**0.4
	density2 = (density2 - np.mean(density2))*alpha

	density2 = 1/(1 + np.exp(-1*density2))
	print(np.min(density2), np.max(density2), np.mean(density2))

	plt.scatter(y_age, pred_y_age, c=density2, cmap="hot", alpha=0.75, s=400, edgecolor='black', linewidth=3) #color='k', edgecolor='black')
	# plt.scatter(y_age, pred_y_age, cmap="k", alpha=0.95, s=200) #color='k', edgecolor='black')
	# cntr1 = ax.contourf(mX, mY, density, alpha=0.35, levels=14, cmap="afmhot_r")

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
		ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}, mae={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error, mae_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
		################################print best fit#############################################
		A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
		w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
			
		y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
		plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='rebeccapurple')
	else:
		corr, pval = png.circ_corrcc(np.squeeze(y_age), np.squeeze(pred_y_age), correction_uniform=True)
		mse_error = get_angle(np.squeeze(y_age), np.squeeze(pred_y_age))
		#print(mse_error, mae_error)
		ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}".format(str(np.round(np.abs(corr), 3)), np.maximum(np.round(pval, 3), 0.001), str(np.round(mse_error, 2))), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
	
	################################print y=x##################################################
	plt.plot(yex, yex, linestyle = 'dashed', linewidth=8, zorder=8, color='k')
	
		
	#plt.title("r= {}, p= {}".format(np.round(corr['r'][0], 2), np.round(corr['p-val'][0], 3)))
	plt.savefig(saveloc, dpi=300, bbox_inches="tight")
	plt.close()


def make_percdata_perf(self_model, finetune, cdist, addon='(in pix)', title_="Large Grid Dataset", saveloc="./Results/ps_eegeyeNet/r_percentage/perf_perc.png"):
	#0.25, 0.5, 0.75, 1.0
	fig, ax = plt.subplots(figsize=(16, 12))
	
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(5)
	ax.spines['left'].set_linewidth(5)

	
	plt.xlabel("Fraction of Training Data", fontsize=45, fontname="Myriad Pro")
	plt.ylabel("RMSE "+addon, fontsize=45, fontname="Myriad Pro")
	plt.title(title_, fontsize=55, fontname="Myriad Pro")
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=40)
	plt.rc('xtick', labelsize=40)
	plt.rc('ytick', labelsize=40)
	#plt.locator_params(axis='both', nbins=6)
	ax.tick_params(width=8)
	


	x = [i for i in range(4)]
	plt.plot(x, self_model, color='skyblue', linewidth=6, label='Self Model')
	plt.plot(x, finetune, color='peachpuff', linewidth=6, label='Finetune (Base Nodel: VS)')
	plt.plot(x, cdist, color='firebrick', linewidth=6, label='CRAFT (Base Nodel: VS)')

	ax.set_xticks([0, 1, 2, 3]) 
	ax.set_xticklabels([0.25, 0.50, 0.75, 1.0]) 
	plt.legend(loc='lower left', fontsize=30)

	plt.savefig(saveloc, dpi=300, bbox_inches="tight")
	plt.close()

def make_plots():

	x = np.linspace(-4.5, 4.5, 1000)
	y = 0.75*x + 1.5*np.random.normal(size=(1000,))
	# print(x.shape,y.shape)

	plot_predictions(y, x, "./Results/paper_plots/build/sample_gmm.png", title_="Test Scatter", _min_use=-5.0, _max_use=5.0)

def print_kseed_results(rmse, r, model_names, saveloc):
	fptr = open(saveloc, 'w')
	for i in range(len(model_names)):
		rmse_i = rmse[i]
		r_i = r[i]
		print("Model: {}".format(model_names[i]), file=fptr)
		print("rmse = {} \u00B1 {} \nr = {} \u00B1 {}".format(np.round(np.mean(rmse_i), 2), np.round(np.std(rmse_i)/np.sqrt(len(rmse_i)), 3), np.round(np.mean(r_i), 2), np.round(np.std(r_i)/np.sqrt(len(r_i)), 3)), file=fptr)
		print("____________________________________________________________________________", file=fptr)
	fptr.close()

def make_percdata_perf_with_error_bars(xtick_labs, supervised, craft, location='lower left', addon='(in pix)', y_label="", title_="CRAFT as a semisupervised tool", saveloc="./Results/iclr/final_results/perf_perc_semisup.png"):
	#0.25, 0.5, 0.75, 1.0
	fig, ax = plt.subplots(figsize=(16, 12))
	

	supervised_mu = np.flip(np.mean(supervised, axis=1))
	supervised_se = np.flip(np.std(supervised, axis=1)/np.sqrt(5))

	craft_mu = np.flip(np.mean(craft, axis=1))
	craft_se = np.flip(np.std(craft, axis=1)/np.sqrt(5))

	improvement = (supervised - craft)/supervised
	improvement_mu = np.flip(np.mean(improvement, axis=1))
	improvement_se = np.flip(np.std(improvement, axis=1)/np.sqrt(5))
	# improvement_se = np.flip(np.sqrt((np.std(craft, axis=1)**2 + np.std(supervised, axis=1)**2)/(5 * supervised_mu**2)))

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(5)
	ax.spines['left'].set_linewidth(5)

	
	plt.xlabel("$n_{ul} / N$", fontsize=45, fontname="Myriad Pro")
	plt.ylabel(y_label+addon, fontsize=45, fontname="Myriad Pro")
	plt.title(title_, fontsize=45, fontname="Myriad Pro")
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=40)
	plt.rc('xtick', labelsize=40)
	plt.rc('ytick', labelsize=40)
	#plt.locator_params(axis='both', nbins=6)
	ax.tick_params(width=8)
	x = [i for i in range(len(xtick_labs))]
	plt.plot(x, supervised_mu, '--', color='firebrick', linewidth=8, label='Supervised', zorder=-2)
	plt.plot(x, craft_mu, color='firebrick', linewidth=8, label='CRAFT', zorder=-2)

	
	# plt.scatter(x, improvement_mu, color='black', alpha=0.90, s=600, edgecolor='black', linewidth=3)
	plt.scatter(x, supervised_mu, facecolor='none', alpha=0.9, s=1500, edgecolor='black', linewidth=6)
	(_, caps, _) =  plt.errorbar(x, supervised_mu, yerr=supervised_se, color='k', fmt='o', linewidth=6,  markersize=6, capsize=10)
	# (_, caps, _) =  plt.errorbar(x, improvement_mu, yerr=improvement_se, color='k', fmt='o', linewidth=4,  markersize=4, capsize=10)
	for cap in caps:
		cap.set_markeredgewidth(6)
	# plt.plot(x, improvement_mu, '--', color='k', linewidth=4, label='Supervised')


	# print(x)
	plt.scatter(x, craft_mu, color='firebrick', alpha=0.9, s=1500, edgecolor='black', linewidth=6)
	(_, caps, _) =  plt.errorbar(x, craft_mu, yerr=craft_se, color='k', fmt='o', linewidth=6,  markersize=6, capsize=10)
	for cap in caps:
		cap.set_markeredgewidth(6)
	
	# plt.plot(x, supervised_mu, '--', color='firebrick', linewidth=4, label='Supervised')
	# plt.plot(x, craft_mu, color='peachpuff', linewidth=6, label='Finetune (Base Nodel: VS)')
	# plt.scatter(x, supervised_mu, color='white', alpha=0.90, s=1500, edgecolor='black', linewidth=6)
	# (_, caps, _) = plt.errorbar(x, craft_mu, yerr=craft_se, color='k', fmt='o', linewidth=8,  markersize=8, capsize=12)
	# for cap in caps:
	# 	cap.set_markeredgewidth(4)

	plt.locator_params(axis="y", nbins=6)

	ax.set_xticks(np.arange(len(xtick_labs))) 
	ax.set_xticklabels([1 - np.flip(xtick_labs)[i] for i in range(len(xtick_labs))]) 
	plt.legend(loc=location, fontsize=30, frameon=False)
	# fig.subplots_adjust(top=0.8) 
	plt.savefig(saveloc, dpi=300, bbox_inches="tight")
	plt.close()

def make_percdata_perf_with_error_bars2(xtick_labs, supervised, craft, location='lower left', addon='(in pix)', y_label="", title_="CRAFT as a semisupervised tool", saveloc="./Results/iclr/final_results/perf_perc_semisup.png", indi=1.0):
	#0.25, 0.5, 0.75, 1.0
	fig, ax = plt.subplots(figsize=(16, 12))
	

	
	improvement = indi*(supervised - craft)/supervised
	improvement_mu = np.flip(np.mean(improvement, axis=1))
	improvement_se = np.flip(np.std(improvement, axis=1)/np.sqrt(5))
	# improvement_se = np.flip(np.sqrt((np.std(craft, axis=1)**2 + np.std(supervised, axis=1)**2)/(5 * supervised_mu**2)))

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(5)
	ax.spines['left'].set_linewidth(5)

	
	plt.xlabel("$n_{ul} / N$", fontsize=45, fontname="Myriad Pro")
	plt.ylabel(y_label+addon, fontsize=45, fontname="Myriad Pro")
	plt.title(title_, fontsize=45, fontname="Myriad Pro")
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=40)
	plt.rc('xtick', labelsize=40)
	plt.rc('ytick', labelsize=40)
	#plt.locator_params(axis='both', nbins=6)
	ax.tick_params(width=8)
	x = [i for i in range(len(xtick_labs))]
	plt.plot(x, improvement_mu, '--', color='black', linewidth=8, label='Supervised', zorder=-2)
	
	
	plt.scatter(x, improvement_mu, color='black', alpha=0.90, s=1500, edgecolor='black', linewidth=6)
	(_, caps, _) =  plt.errorbar(x, improvement_mu, yerr=improvement_se, color='k', fmt='o', linewidth=6,  markersize=6, capsize=10)
	# (_, caps, _) =  plt.errorbar(x, improvement_mu, yerr=improvement_se, color='k', fmt='o', linewidth=4,  markersize=4, capsize=10)
	for cap in caps:
		cap.set_markeredgewidth(6)
	

	ax.set_xticks(np.arange(len(xtick_labs))) 
	ax.set_xticklabels([1 - np.flip(xtick_labs)[i] for i in range(len(xtick_labs))]) 
	# plt.legend(loc=location, fontsize=30, frameon=False)
	# fig.subplots_adjust(top=0.8) 
	plt.locator_params(axis="y", nbins=6)
	plt.savefig(saveloc, dpi=300, bbox_inches="tight")
	plt.close()


def make_plots():

	x = np.linspace(-4.5, 4.5, 1000)
	y = 0.75*x + 1.5*np.random.normal(size=(1000,))
	# print(x.shape,y.shape)

	plot_predictions(y, x, "./Results/paper_plots/build/sample_gmm.png", title_="Test Scatter", _min_use=-5.0, _max_use=5.0)


def make_table_scatter(rmse_mu, rmse_se, r_mu, r_se, rmse_L, r_L, model_names, colors=None):

	# Plot scatter plot
	fig, ax = plt.subplots(figsize=(20, 13))
	if colors is None:
		colors = cm.rainbow(np.linspace(0, 1, len(r_mu)))
	# colors = ['#1f77b4', 'yellow', '#2ca02c', 'pink', '#9467bd', '#8c564b', 'crimson', 'black']
	 
	plt.scatter(rmse_mu, r_mu, color=colors, alpha=0.95, s=1500, edgecolor='black', linewidth=3)
	 
	# Add labels and title
	# plt.title('MAE vs R values for Different Models')
	plt.xlim([47.5, 65.0])
	plt.xlabel('RMSE (in pixels)', fontsize=60)
	plt.ylabel('R', fontsize=60)
	plt.xticks(fontsize=45)
	plt.yticks(fontsize=45)


	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(5)
	ax.spines['left'].set_linewidth(5)
	
	(_, caps, _) =  plt.errorbar(rmse_mu, r_mu, xerr= rmse_se, yerr=r_se, color='k', fmt='o', linewidth=4,  markersize=15, capsize=10)
	for cap in caps:
		cap.set_markeredgewidth(4)

	# Add model names as annotations
	for i, model in enumerate(models):
		plt.annotate(model, (rmse_mu[i], r_mu[i]), (rmse_L[i], r_L[i]), arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=3, connectionstyle="arc3,rad=.2", shrinkB=20), fontsize=35)
	
	# plt.locator_params(axis="x", nbins=7)
	plt.locator_params(axis="y", nbins=4)
	plt.savefig("./Results/iclr/final_results/performance_models.png", bbox_inches="tight", dpi=300)
	plt.close()
	#plt.show()


def plot_continuous_distribution():
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.stats import norm
	fig, ax = plt.subplots(figsize=(16, 12))
	# Define parameters for 6 Gaussian distributions
	means = [-3, -1, 0, 1, 3, 2]
	stds = [0.5, 0.7, 1, 0.8, 0.6, 0.9]
	weights = [0.3, 0.05, 0.2, 0.25, 0.1, 0.1]  # Weights for each Gaussian


	# Generate x-axis values
	x = np.linspace(-10, 10, 500)

	# Create a figure and axis
	# fig, ax = plt.subplots()
	ax.spines['bottom'].set_linewidth(5)
	mixture_density = np.zeros_like(x)
	for mean, std, weight in zip(means, stds, weights):
		mixture_density += weight * norm.pdf(x, mean, std)

	# Plot the mixture density
	ax.plot(x, mixture_density, linewidth=12, color='black')
	ax.fill_between(x, mixture_density, color='gray', alpha=0.3)

	# Customize the plot
	ax.set_yticks([])  # Remove y-axis ticks
	ax.set_xticks([])  # Remove x-axis ticks
	ax.spines['left'].set_visible(False)  # Hide left spine
	ax.spines['top'].set_visible(False)  # Hide top spine
	ax.spines['right'].set_visible(False)  # Hide right spine
	# ax.spines['bottom'].set_visible(False)  # Hide bottom spine   


	# Set x-axis limits
	ax.set_xlim(-6, 6)

	plt.savefig("./Results/iclr/stats/continous_distr.png", dpi=300, bbox_inches="tight")
	plt.close()


def print_performance(saveloc):
    perc_data = [0.01, 0.02, 0.05, 0.10]
    naive_rmse = np.array([[149.16, 149.13, 149.08], [149.36, 149.66, 149.32], [149.37, 149.38, 149.37], [149.40, 149.41, 149.42]])

    tl_rmse = np.array([[90.31, 96.94, 89.53], [82.73, 84.66, 90.06], [70.00, 68.40, 75.27], [64.26, 61.48, 64.38]])
    tl_r = np.array([[0.76, 0.76, 0.78], [0.78, 0.81, 0.77], [0.86, 0.88, 0.84], [0.88, 0.90, 0.88]])

    tasfar_rmse = np.array([[85.94, 89.18, 84.12], [79.36, 86.81, 79.51], [69.34, 62.04, 71.40], [62.87, 59.48, 63.79]])
    tasfar_r = np.array([[0.76, 0.75, 0.78], [0.79, 0.75, 0.81], [0.85, 0.89, 0.84], [0.88, 0.90, 0.88]])

    datafree_rmse = np.array([[90.52, 93.33, 79.08], [79.04, 85.78, 75.52], [71.42, 69.13, 70.96], [70.15, 61.45, 66.81]])
    datafree_r = np.array([[0.78, 0.78, 0.84], [0.82, 0.83, 0.85], [0.87, 0.88, 0.87], [0.89, 0.90, 0.89]])

    craft_rmse = np.array([[93.27, 85.22, 75.01], [73.64, 80.38, 74.82], [69.40, 64.00, 68.78], [64.17, 56.77, 63.28]])
    craft_r = np.array([[0.75, 0.82, 0.86], [0.85, 0.82, 0.85], [0.89, 0.90, 0.88], [0.89, 0.91, 0.89]])

    bbcn_rmse = np.array([[95.96, 109.21, 94.22], [87.32, 93.61, 89.37], [81.54, 77.94, 77.79], [69.35, 71.15, 79.89]])
    bbcn_r = np.array([[0.75, 0.74, 0.79], [0.80, 0.82, 0.83], [0.83, 0.86, 0.84], [0.88, 0.87, 0.84]])

    mixup_rmse = np.array([[138.81, 132.70, 135.59], [124.97, 130.24, 123.20], [109.95, 117.37, 110.32], [102.14, 98.19, 105.97]])
    mixup_r = np.array([[0.51, 0.52, 0.41], [0.64, 0.58, 0.61], [0.67, 0.64, 0.79], [0.74, 0.74, 0.73]])

    fptr = open(saveloc+"results_summary_sacc_2.txt", 'w')
    
    print("EEGNet-LSTM", file=fptr)
    print("------------------------------------------------", file=fptr)
    for i in range(len(perc_data)):
        print("Naive", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(naive_rmse[i]), 2), np.round(np.std(naive_rmse[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
        print("TL", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tl_rmse[i]), 2), np.round(np.std(tl_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(tl_r[i]), 2), np.round(np.std(tl_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
        print("TASFAR", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tasfar_rmse[i]), 2), np.round(np.std(tasfar_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(tasfar_r[i]), 2), np.round(np.std(tasfar_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
        print("DataFree", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(datafree_rmse[i]), 2), np.round(np.std(datafree_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(datafree_r[i]), 2), np.round(np.std(datafree_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
        print("CRAFT", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(craft_rmse[i]), 2), np.round(np.std(craft_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(craft_r[i]), 2), np.round(np.std(craft_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
        print("BBCN", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(bbcn_rmse[i]), 2), np.round(np.std(bbcn_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(bbcn_r[i]), 2), np.round(np.std(bbcn_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)   
        print("Progressive Mixup: ", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(mixup_rmse[i]), 2), np.round(np.std(mixup_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(mixup_r[i]), 2), np.round(np.std(mixup_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
        print("-------------------------------------------------------------------------", file=fptr)

	# tl_rmse = np.array([[106.29, 104.35, 103.73], [89.49, 85.60, 87.54], [79.88, 77.66, 77.81], [73.33, 71.66, 74.42]])
	# tl_r = np.array([[0.70, 0.70, 0.71], [0.75, 0.78, 0.78], [0.80, 0.82, 0.82], [0.84, 0.85, 0.84]])

	# tasfar_rmse = np.array([[128.59, 133.77, 134.26], [87.25, 81.90, 87.60], [76.57, 74.74, 78.79], [71.53, 73.07, 71.83]])
	# tasfar_r = np.array([[0.71, 0.73, 0.72], [0.75, 0.80, 0.80], [0.83, 0.84, 0.81], [0.85, 0.86, 0.85]])

	# datafree_rmse = np.array([[97.68, 92.57, 87.73], [80.47, 82.33, 78.75], [75.44, 73.60, 72.58], [69.79, 68.85, 70.59]])
	# datafree_r = np.array([[0.74, 0.75, 0.78], [0.80, 0.83, 0.83], [0.83, 0.85, 0.85], [0.87, 0.87, 0.88]])

	# craft_rmse = np.array([[112.72, 104.83, 100.36], [90.45, 85.35, 85.85], [77.18, 74.24, 72.19], [69.44, 67.50, 71.63]])
	# craft_r = np.array([[0.72, 0.75, 0.73], [0.78, 0.82, 0.80], [0.83, 0.85, 0.84], [0.86, 0.86, 0.85]])

	# print("PyrCNN", file=fptr)
	# print("------------------------------------------------", file=fptr)
	# for i in range(len(perc_data)):
	# 	print("TL", file=fptr)
	# 	print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tl_rmse[i]), 2), np.round(np.std(tl_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(tl_r[i]), 2), np.round(np.std(tl_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
	# 	print("TASFAR", file=fptr)
	# 	print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tasfar_rmse[i]), 2), np.round(np.std(tasfar_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(tasfar_r[i]), 2), np.round(np.std(tasfar_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
	# 	print("DataFree", file=fptr)
	# 	print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(datafree_rmse[i]), 2), np.round(np.std(datafree_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(datafree_r[i]), 2), np.round(np.std(datafree_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
	# 	print("CRAFT", file=fptr)
	# 	print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(craft_rmse[i]), 2), np.round(np.std(craft_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(craft_r[i]), 2), np.round(np.std(craft_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)		 
	# 	print("-------------------------------------------------------------------------", file=fptr)
	# fptr.close()


if __name__ == '__main__':
	print_performance("./Results/rebuttal_final/")
	# rmse_mu = [63.32, 63.84, 60.95, 54.12, 51.47, 50.66, 48.66]
	# rmse_err = [0.82, 0.75, 0.58, 0.82, 0.63, 0.40, 0.12]
	# r_mu = [0.91, 0.91, 0.91, 0.93, 0.93, 0.93, 0.94]
	# r_err = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.000001]
	
	# rmse_L = [60.14, 62.10, 55.50, 55.72, 51.57, 47.80, 50.00]
	# r_L = [0.915, 0.919, 0.907, 0.935, 0.923, 0.920, 0.94]
	# models = ["Pyr-CNN", "Pyr-CNN+FT", "Pyr-CNN+CRAFT", "EEGNet-LSTM", "EEGNet-LSTM+FT", "EEGNet-LSTM+CRAFT", "EEGNet-LSTM+CRAFT+Ens."]
	# colors = ['lemonchiffon', 'palegreen', 'royalblue', 'lemonchiffon', 'palegreen', 'royalblue', 'crimson']

	# make_table_scatter(rmse_mu, rmse_err, r_mu, r_err, rmse_L, r_L, models, colors)
# 	#semi supervised results
	# plot_continuous_distribution()
	
	#---------------------------------------------------eegnet-lstm on vs dataset-----------------------------------------------------
	# ratio = [0.02, 0.05, 0.10, 0.25, 0.5, 0.75]
	# rmse_sup = np.array([[148.06, 147.44, 161.49, 151.64, 141.66], 
	# 				 [106.97, 98.63, 101.39, 105.65, 103.35],
	# 				 [90.51, 83.71, 89.58, 91.94, 87.20],
	# 				 [74.02, 72.81, 73.52, 74.43, 74.43],
	# 				 [70.96, 71.13, 69.84, 72.83, 70.90],
	# 				 [69.60, 70.67, 70.53, 70.88, 72.27]])

	# r_sup = np.array([[0.45, 0.43, 0.30, 0.40, 0.51],
	# 				 [0.78, 0.82, 0.81, 0.79, 0.81],
	# 				 [0.86, 0.87, 0.87, 0.86, 0.85],
	# 				 [0.89, 0.89, 0.89, 0.89, 0.89],
	# 				 [0.90, 0.90, 0.90, 0.89, 0.90],
	# 				 [0.90, 0.90, 0.90, 0.90, 0.90]])
	
	# rmse_craft = np.array([[104.76, 109.19, 100.93, 112.12, 102.13],
	# 				 [87.43, 98.01, 93.90, 93.84, 85.78],
	# 				 [81.52, 84.59, 80.20, 85.95, 80.74],
	# 				 [75.48, 74.80, 75.70, 73.07, 76.86],
	# 				 [71.79, 72.53, 71.30, 70.85, 72.50],
	# 				 [69.12, 69.83, 70.61, 67.79, 69.19]])

	# r_craft = np.array([[0.80, 0.78, 0.82, 0.76, 0.80],
	# 				 [0.87, 0.83, 0.84, 0.81, 0.85],
	# 				 [0.88, 0.87, 0.87, 0.86, 0.87],
	# 				 [0.89, 0.89, 0.89, 0.90, 0.88],
	# 				 [0.90, 0.90, 0.90, 0.90, 0.90],
	# 				 [0.91, 0.90, 0.90, 0.91, 0.91]])

	# make_percdata_perf_with_error_bars2(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_vs_eegnet_rmse.png")
	# make_percdata_perf_with_error_bars(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="RMSE (in pix)", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_vs_eegnet_relChange_rmse.png")

	# make_percdata_perf_with_error_bars2(ratio, r_sup, r_craft, location='upper right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_vs_eegnet_r.png", indi=-1)
	# make_percdata_perf_with_error_bars(ratio, r_sup, r_craft, location='upper right', addon='', y_label="R", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_vs_eegnet_relChange_r.png")



	# # WM data set Eegnet-lstm on wm dataset using CRAFT + TL-------------------------------------------------------------------------
	# #------------------------------------------------- eegnet-lstm on wm dataset ------------------------------------------------------
	# ratio = [0.05, 0.10, 0.25, 0.5, 0.75]
	# rmse_sup = np.array([[180.43, 179.07, 179.53, 178.21, 182.93],
	# 				 [171.79, 166.58, 171.55, 165.10, 167.88],
	# 				 [158.39, 150.11, 145.36, 156.03, 147.91],
	# 				 [144.08, 140.71, 133.05, 138.36, 135.86],
	# 				 [132.16, 131.31, 123.41, 118.92, 130.69]])

	# r_sup = np.array([[0.38, 0.34, 0.34, 0.36, 0.32],
	# 				[0.40, 0.50, 0.39, 0.45, 0.44],
	# 				[0.56, 0.59, 0.62, 0.53, 0.59],
	# 				[0.68, 0.66, 0.69, 0.66, 0.66],
	# 				[0.71, 0.69, 0.74, 0.75, 0.70]])
	
	# rmse_craft = np.array([[178.43, 177.13, 181.42, 179.05, 182.10],
	# 				 [172.20, 169.77, 175.45, 172.80, 169.69],
	# 				 [174.09, 153.44, 144.60, 160.15, 146.64],
	# 				 [128.20, 132.18, 139.80, 135.99, 129.86],
	# 				 [117.98, 125.94, 128.60, 122.77, 115.17]])

	# r_craft = np.array([[0.34, 0.37, 0.31, 0.32, 0.29],
	# 				 [0.41, 0.45, 0.37, 0.36, 0.42],
	# 				 [0.56, 0.61, 0.64, 0.53, 0.63],
	# 				 [0.70, 0.71, 0.66, 0.69, 0.72],
	# 				 [0.75, 0.72, 0.71, 0.74, 0.76]])

	# make_percdata_perf_with_error_bars2(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_tl_wm_eegnet_rmse.png")
	# make_percdata_perf_with_error_bars(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="RMSE (in pix)", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_tl_wm_eegnet_relChange_rmse.png")

	# make_percdata_perf_with_error_bars2(ratio, r_sup, r_craft, location='upper right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_tl_wm_eegnet_r.png", indi=-1)
	# make_percdata_perf_with_error_bars(ratio, r_sup, r_craft, location='upper right', addon='', y_label="R", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_tl_wm_eegnet_relChange_r.png")



	

	# #------------------------------------------------- eegnet-lstm on wm dataset ------------------------------------------------------

	# rmse_sup = np.array([[176.86, 184.02, 183.07, 179.81, 180.67],
	# 				 [175.19, 177.75, 175.45, 179.81, 180.53],
	# 				 [177.25, 175.82, 176.81, 179.32, 177.91],
	# 				 [185.05, 167.79, 174.39, 176.80, 187.67],
	# 				 [145.76, 139.62, 142.97, 137.06, 147.56]])

	# r_sup = np.array([[0.32, 0.31, 0.27, 0.28, 0.27],
	# 				 [0.34, 0.31, 0.34, 0.26, 0.26],
	# 				 [0.32, 0.38, 0.35, 0.35, 0.30],
	# 				 [0.51, 0.57, 0.52, 0.45, 0.49],
	# 				 [0.66, 0.69, 0.68, 0.71, 0.66]])
	
	# rmse_craft = np.array([[187.86, 179.30, 187.50, 189.26, 187.43],
	# 				 [182.38, 179.59, 180.73, 178.71, 178.39],
	# 				 [178.90, 181.14, 176.89, 179.29, 181.22],
	# 				 [176.15, 164.24, 169.82, 157.82, 178.92],
	# 				 [125.95, 139.45, 130.15, 128.45, 133.00]])

	# r_craft = np.array([[0.19, 0.28, 0.28, 0.19, 0.21],
	# 				 [0.25, 0.32, 0.29, 0.27, 0.33],
	# 				 [0.34, 0.31, 0.34, 0.34, 0.29],
	# 				 [0.55, 0.61, 0.62, 0.61, 0.54],
	# 				 [0.75, 0.71, 0.73, 0.75, 0.73]])

	# make_percdata_perf_with_error_bars2(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_wm_eegnet_rmse.png")
	# make_percdata_perf_with_error_bars(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="RMSE (in pix)", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_wm_eegnet_relChange_rmse.png")

	# make_percdata_perf_with_error_bars2(ratio, r_sup, r_craft, location='upper right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_wm_eegnet_r.png", indi=-1)
	# make_percdata_perf_with_error_bars(ratio, r_sup, r_craft, location='upper right', addon='', y_label="R", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_notl_wm_eegnet_relChange_r.png")


	#------------------------------------------------- eegnet-lstm on large grid ------------------------------------------------------
# 	models = ['EEGNet-LSTM (supervised)', 'EEGNet-LSTM + CRAFT']

	# rmse_sup = np.array([[87.80, 103.28, 91.32, 98.93, 91.83],
	# 				 [82.87, 81.69, 83.53, 84.94, 80.65],
	# 				 [68.25, 66.60, 67.46, 64.06, 64.35],
	# 				 [57.90, 55.95, 57.37, 60.05, 60.53],
	# 				 [57.82, 54.67, 53.62, 54.82, 55.91]])

	# r_sup = np.array([[0.79, 0.77, 0.80, 0.80, 0.77],
	# 				 [0.85, 0.84, 0.85, 0.84, 0.84],
	# 				 [0.89, 0.89, 0.90, 0.90, 0.90],
	# 				 [0.92, 0.92, 0.91, 0.90, 0.91],
	# 				 [0.92, 0.92, 0.93, 0.93, 0.92]])

# # 	print_kseed_results(rmse_sup, r_sup, ratio, "./Results/iclr/final_results/perc_supervised.txt")
	
	# rmse_craft = np.array([[78.39, 73.83, 78.13, 75.48, 80.98],
	# 				 [65.71, 67.52, 69.91, 72.50, 65.62],
	# 				 [63.10, 60.77, 65.21, 59.62, 60.29],
	# 				 [56.06, 52.81, 56.15, 53.89, 57.13],
	# 				 [55.31, 55.06, 50.87, 53.09, 52.67]])

	# r_craft = np.array([[0.86, 0.86, 0.83, 0.86, 0.84],
	# 				 [0.89, 0.88, 0.88, 0.86, 0.89],
	# 				 [0.91, 0.92, 0.90, 0.90, 0.91],
	# 				 [0.92, 0.93, 0.92, 0.93, 0.92],
	# 				 [0.92, 0.93, 0.93, 0.93, 0.93]])
	
# 	print_kseed_results(rmse_craft, r_craft, ratio, "./Results/iclr/final_results/perc_craft.txt")
	
	# make_percdata_perf_with_error_bars2(ratio, r_sup, r_craft, location='upper right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_perc_semisup_r_impr_exp_largemarker_0.1.png")
	# make_percdata_perf_with_error_bars(ratio, r_sup, r_craft, location='upper right', addon='', y_label="R", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_perc_semisup_r_impr2_exp_largemarker_0.1.png")
	# make_percdata_perf_with_error_bars2(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="Relative Change", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_perc_semisup_r_impr_exp_largemarker.png")
	# make_percdata_perf_with_error_bars2(ratio, rmse_sup, rmse_craft, location='lower right', addon='', y_label="RMSE (in pix)", title_="Labeled vs (Labeled + Unlabeled)", saveloc="./Results/iclr/final_results/perf_perc_semisup_r_impr2_exp_largemarker.png")
# 	make_percdata_perf_with_error_bars(ratio, r_sup, r_craft, location='lower right', addon="", y_label="R", title_="EEGNet-LSTM: Semi-supervised", saveloc="./Results/iclr/final_results/perf_perc_semisup_r.png")
	# rmse = np.array([[170.14, 170.14, 170.14, 170.14, 170.14],
	# 				 [67.64, 67.64, 67.64, 67.64, 67.64],
	# 				 [79.59, 79.59, 79.59, 79.59, 79.59]])

	# r = np.array([[0.00, 0.00, 0.00, 0.00, 0.00],
	# 			  [0.91, 0.91, 0.91, 0.91, 0.91],
	# 			  [0.87, 0.87, 0.87, 0.87, 0.87]])

	# models = ['Naive Baseline', 'EEGNet-LSTM (self)', 'Spy-CNN (self)']
	# print_kseed_results(rmse, r, models, "./Results/iclr/final_results/VS_res.txt")


	# rmse = np.array([[149.40, 149.40, 149.40, 149.40, 149.40],
	# 				 [57.65, 52.50, 53.28, 53.23, 53.94],
	# 				 [50.82, 50.91, 50.79, 50.82, 50.31],
	# 				 [51.29, 52.54, 51.99, 48.81, 52.70],
	# 				 [49.34, 50.47, 49.49, 49.41, 49.68],
	# 				 [49.73, 51.85, 52.29, 50.89, 51.31], 
	# 				 [49.52, 49.43, 49.12, 49.20, 49.86],
	# 				 [61.62, 61.70, 66.36, 62.45, 64.47],
	# 				 [55.94, 56.59, 55.54, 56.90, 56.90],
	# 				 [63.91, 61.93, 66.91, 63.54, 62.93],
	# 				 [59.03, 58.80, 58.13, 59.45, 58.63],
	# 				 [60.72, 63.28, 61.17, 62.02, 62.14],
	# 				 [56.50, 56.66, 56.99, 56.14, 56.92]])

	# r = np.array([[0.00, 0.00, 0.00, 0.00, 0.00],
	# 			  [0.92, 0.93, 0.93, 0.93, 0.93],
	# 			  [0.94, 0.94, 0.94, 0.94, 0.94],
	# 			  [0.93, 0.93, 0.94, 0.94, 0.93],
	# 			  [0.94, 0.94, 0.94, 0.94, 0.94], 
	# 			  [0.93, 0.93, 0.93, 0.94, 0.94],
	# 			  [0.94, 0.94, 0.94, 0.94, 0.94],
	# 			  [0.90, 0.91, 0.91, 0.91, 0.90],
	# 			  [0.93, 0.93, 0.93, 0.93, 0.93],
	# 			  [0.91, 0.90, 0.90, 0.91, 0.91],
	# 			  [0.92, 0.92, 0.92, 0.92, 0.92],
	# 			  [0.90, 0.91, 0.90, 0.91, 0.90],
	# 			  [0.92, 0.92, 0.92, 0.92, 0.92]])

	# models = ['Naive Baseline', 
	# 		  'EEGNet-LSTM (self)', 'EEGNet-LSTM (self, Ens)', 'EEGNet-LSTM (TL: VS)', 'EEGNet-LSTM (TL: VS, Ens)', 
	# 		  'EEGNet-LSTM (CRAFT: VS)', 'EEGNet-LSTM (CRAFT: VS, Ens)', 'Spy-CNN (self)', 'Spy-CNN (self, Ens)', 
	# 		  'Spy-CNN (TL: VS)', 'Spy-CNN (TL: VS, Ens)', 'Spy-CNN (CRAFT: VS)', 'Spy-CNN (CRAFT: VS, Ens)']
	# print_kseed_results(rmse, r, models, "./Results/iclr/final_results/LG_res.txt")

	# rmse = np.array([[188.18, 188.18, 188.18, 188.18, 188.82],
	# 				 [136.07, 133.20, 141.98, 125.67, 132.13],
	# 				 [126.80, 128.34, 124.68, 126.29, 126.20],
	# 				 [111.33, 117.10, 113.18, 114.29, 110.83],
	# 				 [110.55, 109.18, 109.32, 109.06, 110.17],
	# 				 [118.55, 114.78, 116.03, 115.58, 115.09],
	# 				 [112.56, 111.35, 111.82, 111.90, 111.13], 
	# 				 [141.26, 141.16, 137.41, 146.03, 147.40],
	# 				 [129.15, 129.28, 131.17, 130.78, 130.59],
	# 				 [141.48, 139.12, 142.30, 141.42, 143.09],
	# 				 [132.08, 132.55, 132.51, 132.78, 132.92],
	# 				 [149.12, 147.78, 147.56, 145.95, 146.22],
	# 				 [138.75, 138.47, 138.00, 138.33, 137.67]])

	# r = np.array([[0.00, 0.00, 0.00, 0.00, 0.00],
	# 			  [0.70, 0.72, 0.70, 0.74, 0.73],
	# 			  [0.73, 0.73, 0.74, 0.74, 0.74],
	# 			  [0.78, 0.76, 0.77, 0.76, 0.78],
	# 			  [0.78, 0.79, 0.78, 0.79, 0.78],
	# 			  [0.75, 0.76, 0.76, 0.76, 0.77],
	# 			  [0.77, 0.78, 0.77, 0.77, 0.78],
	# 			  [0.64, 0.63, 0.66, 0.61, 0.64],
	# 			  [0.69, 0.70, 0.68, 0.69, 0.69],
	# 			  [0.62, 0.64, 0.62, 0.65, 0.62],
	# 			  [0.67, 0.67, 0.67, 0.67, 0.67],
	# 			  [0.61, 0.63, 0.60, 0.62, 0.63],
	# 			  [0.65, 0.66, 0.66, 0.65, 0.66]])

	# models = ['Naive Baseline', 
	# 		  'EEGNet-LSTM (self)', 'EEGNet-LSTM (self, Ens)', 'EEGNet-LSTM (TL: VS)', 'EEGNet-LSTM (TL: VS, Ens)', 
	# 		  'EEGNet-LSTM (CRAFT: VS)', 'EEGNet-LSTM (CRAFT: VS, Ens)', 'Spy-CNN (self)', 'Spy-CNN (self, Ens)', 
	# 		  'Spy-CNN (TL: VS)', 'Spy-CNN (TL: VS, Ens)', 'Spy-CNN (CRAFT: VS)', 'Spy-CNN (CRAFT: VS, Ens)']
	# print_kseed_results(rmse, r, models, "./Results/iclr/final_results/WM_res.txt")

	# make_plots()
	# self_model = [69.21, 68.03, 54.50, 54.76] 
	# finetune = [60.08, 63.58, 53.94, 52.32]
	# cdist = [58.92, 58.24, 52.68, 48.34]
	# make_percdata_perf(self_model, finetune, cdist, addon='(in pix)', title_="Large Grid Dataset", saveloc="./Results/ps_eegeyeNet/r_percentage/perf_perc.png")

	# self_model = [182.32, 152.80, 134.93, 127.15] 
	# finetune = [155.21, 137.96, 123.21, 121.25]
	# cdist = [168.52, 124.40, 128.29, 117.00]
	# make_percdata_perf(self_model, finetune, cdist, addon='(in pix)', title_="Working Memory Dataset", saveloc="./Results/ps_eegeyeNet/wm_r_percentage/perf_perc.png")