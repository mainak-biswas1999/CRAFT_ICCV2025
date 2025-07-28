from loader import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc
from eye_dat_prep import *


class EM_Exp_GMM_1d:
	def __init__(self, n_g, n_e, X):
		self.n_g = n_g    #number of gaussians = k1
		self.n_e = n_e    #number of exponentials = k2
		
		self.X = X      #(N, 1)
		# the variational distribution q(z = j | X_i) = q_ji -> dimension (N, k1) - for normal, (N, k2) - for normal
		self.q1 = None
		self.q2 = None
		#dimension (k1, 1) -> k normals, univariate
		self.mus = None
		#dimension (k1, 1) -> k normals, univariate 
		self.sigmas = None
		#lambdas (k2, 1) -> for exponential distribution
		self.lambdas = None
		#multinomial - (k1+k2,)
		self.alphas = None #k1 gaussians
		self.betas = None  #k2 exponentials
		
		#initialize the parameters
		self.initialize_params()
		print("Data dimension: {} \nDim. alphas: {} \nDim. mus: {}, \nDim. sigmas: {} \nDim. lambdas: {} \nDim q1: {} \nDim q2: {}".format(self.X.shape, self.alphas.shape, self.mus.shape, self.sigmas.shape, self.lambdas.shape, self.q1.shape, self.q2.shape))
	
	def initialize_params(self):
		np.random.seed(99)
		#list you want to append
		mus = []
		# for the features randomly set the means of the distributions
		#covariance of the data
		sigma_data = np.std(self.X.ravel())
		
		for i in range(self.X.shape[1]):
			mus.append(np.random.normal(loc=np.mean(self.X[:, i]), scale=np.std(self.X[:, i]), size=(self.n_g, 1)))
			
		#The sigmas of all the normals is set to the covariance of the data 
		self.sigmas = np.repeat(np.expand_dims(sigma_data, axis=(0, 1)), self.n_g, axis=0)
		self.alphas = np.repeat(1./self.n_g, self.n_g)
		#setting uniform probabilities for each datapoint in the variational distribution
		self.q1 = np.repeat(np.expand_dims(self.alphas, axis=0), self.X.shape[0], axis=0)
		self.mus = np.concatenate(mus, axis=1)
		
		self.betas = np.repeat(1./self.n_e, self.n_e)
		#setting uniform probabilities for each datapoint in the variational distribution
		self.q2 = np.repeat(np.expand_dims(self.betas, axis=0), self.X.shape[0], axis=0)
		self.lambdas = np.ones((self.n_e, 1))

	def get_all_normal_values(self, X):
		# get the normal pdf value of all the data points for all the normals in the GMM (N, k1) 
		all_normal_vals = np.zeros((X.shape[0], self.n_g))
		
		for i in range(self.n_g):
			#calculate the normal values for all the data points and a single normal
			for j in range(X.shape[0]):
				exp_term_ji = np.exp(-0.5* ((X[j, 0] - self.mus[i, 0])/self.sigmas[i, 0])**2)
				all_normal_vals[j, i] = (1./(np.sqrt(2*np.pi) * self.sigmas[i, 0])) * exp_term_ji
			
		return all_normal_vals
	
	def get_all_exp_values(self, X):
		# get the exponential pdf value of all the data points for all Exp. (N, k2) 
		all_exp_vals = np.zeros((X.shape[0], self.n_e))
		
		for i in range(self.n_e):
			#calculate the normal values for all the data points and a single normal
			for j in range(X.shape[0]):
				all_exp_vals[j, i] = self.lambdas[i, 0] * np.exp(-self.lambdas[i, 0] * X[j, 0]) * int(X[j, 0] > 0.)
		
		return all_exp_vals
	
	def Expectation(self):
		all_normal_vals = self.get_all_normal_values(self.X)
		all_exp_vals = self.get_all_exp_values(self.X)
		#calculate the variational distribution: dimension of q: (N, k)
		#print(np.expand_dims(self.alphas, axis=0).shape)
		q1_alpha_prod = np.expand_dims(self.alphas, axis=0) * all_normal_vals 
		q2_alpha_prod = np.expand_dims(self.betas, axis=0) * all_exp_vals
		#normalize all the qs by the sum across k normals
		norm_val = np.sum(q1_alpha_prod, axis=1, keepdims=True) + np.sum(q2_alpha_prod, axis=1, keepdims=True)
		self.q1 = q1_alpha_prod / norm_val
		self.q2 = q2_alpha_prod / norm_val

	def Maximization(self):
		#set the variational distribution q
		self.alphas = np.mean(self.q1, axis=0)
		self.betas = np.mean(self.q2, axis=0)
		#print(np.sum(self.alphas))
		norm_term = np.expand_dims(self.X.shape[0] * self.alphas, axis=1)  #converting it to k1x1
		self.mus = (self.q1.T @ self.X) / norm_term
		#sigmas = np.zeros(self.sigmas.shape)
		for i in range(self.n_g):
			#X has examples in the rows
			mu_i = np.expand_dims(self.mus[i, 0], axis=0)
			#\|q (X-u)
			X_tilde = (self.X - mu_i) * np.sqrt(np.expand_dims(self.q1[:, i], axis=1))
			self.sigmas[i, 0] = np.sqrt((X_tilde.T @ X_tilde)[0, 0] / norm_term[i, 0]) 
		#exponentials - 
		norm_term2 = np.expand_dims(self.X.shape[0] * self.betas, axis=1)  #converting it to k2x1
		self.lambdas = norm_term2 / (self.q2.T @ self.X) 
		 

	def run_EM(self, max_iter=100):
		for i in range(max_iter):
			self.Expectation()
			self.Maximization()
	
	def return_P(self, X):
		P_normal = self.get_all_normal_values(X)
		P_exp = self.get_all_exp_values(X)
		P =  P_exp @ np.expand_dims(self.betas, axis=1) +  P_normal @ np.expand_dims(self.alphas, axis=1)
		return P



def plot_histo(v, density_est_x, density_est_y, nbins=1000, xlabel_="Velocity in pix/s (in logscale)", y_label="Density", title_="Working memory eye-tracking Velocity distribution", saveloc='./Results/EEGNet_LSTM_WM/DPos/eye_vel.png'):
	#plot the histogram of velocities
	ignore_nan_v = np.ravel(v)
	
	plt.hist(ignore_nan_v, bins=nbins, density=True)
			
	plt.plot(density_est_x, density_est_y, 'r')
	print(auc(density_est_x, density_est_y))


	plt.xlabel(xlabel_, fontsize=16)
	plt.ylabel(y_label, fontsize=16)
	plt.title(title_, fontsize=16)
	plt.savefig(saveloc)
	plt.close()


def min_cross_val_error(data, fname):
	np.random.seed(25)
	test_list = np.random.choice(data.shape[0], int(0.2*data.shape[0]), replace=False)
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

def learn_py(type_='r', in_='pix'):

	data_loader = integrated_data(type_)
	_, target_train_y, _, target_test_y, _, target_val_y = data_loader.get_data_target()
	_, source_train_y, _, source_test_y, _, source_val_y = data_loader.get_data_source()
	
	source_y = source_train_y
	# source_y = np.concatenate([source_train_y, source_val_y], axis=0)
	target_y = target_train_y
	# target_y = np.concatenate([target_train_y, target_val_y], axis=0)

	# e_gmm = EM_Exp_GMM_1d(2, 2, target_train_y+1)
	# e_gmm.run_EM()
	# test_fit_y = np.expand_dims(np.linspace(np.min(target_train_y+1)-0.1, np.max(target_train_y+1)+0.1, 1000), axis=-1)
	# P_vals = e_gmm.return_P(test_fit_y)

	# plot_histo(target_train_y, test_fit_y-1, P_vals, 100, "r (in pix)", "Density", "del r", "./Results/pilot_models/stats/r_wm_train_fit_e_gmm.png")
	# print(P_vals.shape)
	# min_cross_val_error(source_train_y, "./Results/ps_eegeyeNet/stats/ps_dset_fit_gmm_3_{}.png".format(type_))
	# min_cross_val_error(target_train_y, "./Results/ps_eegeyeNet/stats/ddir_dset_fit_gmm_3_{}.png".format(type_))

	#fit plotting
	if type_ == 'r':
		p_y_gmm = GaussianMixture(n_components=14, random_state=0).fit(data_loader.inv_scale(target_y))
		p_y_gmm2 = GaussianMixture(n_components=8, random_state=0).fit(data_loader.inv_scale(source_y))

	else:
		p_y_gmm = GaussianMixture(n_components=11, random_state=0).fit(data_loader.inv_scale(target_y))
		p_y_gmm2 = GaussianMixture(n_components=6, random_state=0).fit(data_loader.inv_scale(source_y))

		
	test_fit_y = np.expand_dims(np.linspace(np.min(target_y)-0.1, np.max(target_y)+0.1, 1000), axis=-1)
	test_fit_py = np.exp(p_y_gmm.score_samples(data_loader.inv_scale(test_fit_y)))
	plot_histo(data_loader.inv_scale(target_y), data_loader.inv_scale(test_fit_y), test_fit_py, 100, "{} (in {})".format(type_, in_), "Density", "del {}".format(type_), "./Results/ps_eegeyeNet/stats/{}_eye_train_fit_gmm_train_val.png".format(type_))
	plot_histo(data_loader.inv_scale(target_test_y), data_loader.inv_scale(test_fit_y), test_fit_py, 100, "{} (in {})".format(type_, in_), "Density", "del {}".format(type_), "./Results/ps_eegeyeNet/stats/{}_eye_train_fit_gmm_test2.png".format(type_))
	
	test_fit_y2 = np.expand_dims(np.linspace(np.min(source_y)-0.1, np.max(source_y)+0.1, 1000), axis=-1)
	test_fit_py2 = np.exp(p_y_gmm2.score_samples(data_loader.inv_scale(test_fit_y2)))
	plot_histo(data_loader.inv_scale(source_y), data_loader.inv_scale(test_fit_y2), test_fit_py2, 100, "{} (in {})".format(type_, in_), "Density", "del {}".format(type_), "./Results/ps_eegeyeNet/stats/{}_ps_train_fit_gmm_train_val.png".format(type_))
	plot_histo(data_loader.inv_scale(source_test_y), data_loader.inv_scale(test_fit_y2), test_fit_py2, 100, "{} (in {})".format(type_, in_), "Density", "del {}".format(type_), "./Results/ps_eegeyeNet/stats/{}_ps_train_fit_gmm_test2.png".format(type_))

def learn_py_robots():
	eye_dat = get_eye_only_npz()
	print(eye_dat.shape)
	 

	min_cross_val_error(np.expand_dims(eye_dat[:, 0], axis=-1), "./Results/eeg_robots_drift_1s/stats/r_gmm_cv_error_fit.png")
	min_cross_val_error(np.expand_dims(eye_dat[:, 1], axis=-1), "./Results/eeg_robots_drift_1s/stats/theta_gmm_cv_error_fit.png")
	
	r_dat = np.expand_dims(eye_dat[:, 0], axis=-1)
	theta_dat = np.expand_dims(eye_dat[:, 1], axis=-1)
	
	# e_gmm = EM_Exp_GMM_1d(1, 3, r_dat)
	# e_gmm.run_EM()
	# test_fit_y = np.expand_dims(np.linspace(np.min(r_dat)-0.1, np.max(r_dat)+0.1, 1000), axis=-1)
	# P_vals = e_gmm.return_P(test_fit_y)

	# plot_histo(r_dat, test_fit_y, P_vals, 100, "r (in pix)", "Density", "del r", "./Results/eeg_gaze_robots/stats/exp_gmm_fit.png")

	p_y_gmm_r = GaussianMixture(n_components=5, random_state=0).fit(r_dat)
	p_y_gmm_theta = GaussianMixture(n_components=5, random_state=0).fit(theta_dat)
	# print(np.exp(p_y_gmm_r.score_samples(r_dat))[0:100])

	test_fit_y = np.expand_dims(np.linspace(np.min(r_dat)-0.1, np.max(r_dat)+0.1, 1000), axis=-1)
	test_fit_py = np.exp(p_y_gmm_r.score_samples(test_fit_y))
	plot_histo(r_dat, test_fit_y, test_fit_py, 100, "r (in pix)", "Density", "del r", "./Results/eeg_robots_drift_1s/stats/r_gmm_fit.png")

	test_fit_y2 = np.expand_dims(np.linspace(np.min(theta_dat)-0.1, np.max(theta_dat)+0.1, 1000), axis=-1)
	test_fit_py2 = np.exp(p_y_gmm_theta.score_samples(test_fit_y2))
	plot_histo(theta_dat, test_fit_y2, test_fit_py2, 100, "theta (in pix)", "Density", "del theta", "./Results/eeg_robots_drift_1s/stats/theta_gmm_fit.png")

# learn_py_robots()
learn_py()
# learn_py('theta', 'radians')