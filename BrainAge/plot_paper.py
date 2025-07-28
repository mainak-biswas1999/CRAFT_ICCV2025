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

	plt.scatter(y_age, pred_y_age, c=density2, cmap="hot", alpha=0.75, s=400, edgecolor='black', linewidth=3) #color='k', edgecolor='black')


	# plt.scatter(y_age, pred_y_age, c=p_y_gmm.score_samples(data), cmap="afmhot_r", alpha=0.95, s=400, edgecolor='black', linewidth=3) #color='k', edgecolor='black')
	# plt.scatter(y_age, pred_y_age, cmap="k", alpha=0.95, s=200) #color='k', edgecolor='black')
	# cntr1 = ax.contourf(mX, mY, density, alpha=0.33, levels=14, cmap="binary")

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
		plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='plum')
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
	# plt.xlim([5.86, 6.50])
	# plt.ylim([0.59, 0.70])
	plt.xlabel('RMSE (in years)', fontsize=60)
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
	
	plt.locator_params(axis="x", nbins=7)
	plt.locator_params(axis="y", nbins=5)
	plt.savefig("./Results/iclr/stats/performance_models_subsampling_eecs.png", bbox_inches="tight", dpi=300)
	plt.close()
	#plt.show()


def two_sample_t_test(x_mu, x_se, y_mu, y_se, n):
	from scipy.stats import ttest_ind_from_stats
	t = ttest_ind_from_stats(mean1=x_mu, std1=(x_se * n/np.sqrt(n-1)) , nobs1=n, mean2=y_mu, std2=(x_se * n/np.sqrt(n-1)), nobs2=n)
	print(t)

if __name__ == '__main__':
	#pyr cnn tl vs craft
	# two_sample_t_test(63.84, 0.75, 60.95, 0.58, 5)  #*
	# two_sample_t_test(116.36, 0.89, 113.02, 1.90, 5) #*
	# two_sample_t_test(51.47, 0.63, 50.66, 0.40, 5)  #not significant
	# print("-------------------------------------------------------")
	# two_sample_t_test(6.14, 0.03, 6.04, 0.02, 5)  # *
	# two_sample_t_test(7.34, 0.04, 7.14, 0.06, 5)  # *
	# two_sample_t_test(0.66, 0.003, 0.68, 0.003, 5)
	# rmse_mu = [6.26, 6.14, 6.04, 5.94, 5.90]
	# rmse_err = [0.00, 0.03, 0.02, 0.04, 0.01]
	# r_mu = [0.63, 0.66, 0.68, 0.68, 0.69]
	# r_err = [0.000, 0.003, 0.003, 0.004, 0.002]
	
	# rmse_L = [6.21, 6.08, 6.09, 5.92,  5.92]
	# r_L = [0.63, 0.652, 0.683, 0.67,  0.688]
	
	# rmse_mu = [7.34, 7.14, 6.24, 6.15]
	# rmse_err = [0.04, 0.06, 0.06, 0.01]
	
	# r_mu = [0.57, 0.60, 0.68, 0.70]
	# r_err = [0.006, 0.014, 0.007, 0.002]
	
	# # rmse_L = [7.04, 6.84, 6.25, 6.30]
	# # r_L = [0.565, 0.62, 0.65, 0.70]
	# models = ["SFCN", "SFCN+FT", "SFCN+CRAFT", "SFCN+CRAFT+RADC", "SFCN+CRAFT+RADC+Ens."]
	# colors = ['lemonchiffon', 'palegreen', 'royalblue', 'navy', 'crimson']
	# make_table_scatter(rmse_mu, rmse_err, r_mu, r_err, rmse_L, r_L, models, colors)
	rmse_L = [8.41, 8.00, 8.62, 8.60, 8.70, 7.80]
	r_L = [0.43, 0.25, 0.25, 0.32, 0.45, 0.38]
	
	rmse_mu =  [8.38, 8.19, 8.53, 8.57, 8.58, 7.98]
	rmse_err = [0.27, 0.15, 0.16, 0.22, 0.25, 0.25]
	
	r_mu =  [0.41, 0.32, 0.23, 0.39, 0.41, 0.45]
	r_err = [0.05, 0.02, 0.02, 0.04, 0.03, 0.03]

	models = ["TL", "MixUp", "BBCN", "TASFAR", "DataFree", "CRAFT"]
	# colors = ['silver', 'darkorchid', 'mediumseagreen', 'goldenrod', 'lightskyblue', 'firebrick']
	colors = ['silver', 'pink', 'mediumseagreen', 'goldenrod', 'lightskyblue', 'firebrick']
	make_table_scatter(rmse_mu, rmse_err, r_mu, r_err, rmse_L, r_L, models, colors)



# if __name__ == '__main__':
# 	make_plots()