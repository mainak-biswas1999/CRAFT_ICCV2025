from read_EEGEyenet import *
from eye_dat_prep import *
from read_deepak_data import *
from sklearn.mixture import GaussianMixture


class Rescaling:
    def __init__(self, __type__='linear', a=-1.0, b=1.0):
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
        
    
    def set_params(self, eye_data):
        if self.type == 'linear':
            self.vmin = np.min(eye_data[:, 0])
            self.vmax = np.max(eye_data[:, 0])
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
        #    sys.exit()
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


class EM_Exp_GMM_1d:
	def __init__(self, n_g, n_e, X):
		self.n_g = n_g	#number of gaussians = k1
		self.n_e = n_e	#number of exponentials = k2
		
		self.X = X	  #(N, 1)
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


class integrated_data:
	def __init__(self, type_='r', target_dataset='DDir', perc_data_target=1.0):
		#type_ --> r, theta, and both
		#will be deepak's dataset
		self.perc_data_target = perc_data_target
		self.source_train_list = None
		self.source_test_list = None
		self.source_X = None
		self.source_y = None
		self.target_dataset = target_dataset
		self.source_scale_obj_eeg = Rescaling(__type__='z-score')
		#will be eeg-eyenet
		self.target_train_list = None
		self.target_train_list_full = None
		self.target_test_list = None
		self.target_X = None
		self.target_y = None
		self.target_scale_obj_eeg = Rescaling(__type__='z-score')
		#this will scale the data
		self.type_ = type_
		if self.type_ == 'both':
			self.scale_obj = Rescaling(__type__='linear2', a=-1.0, b=1.0)
		else:
			self.scale_obj = Rescaling(__type__='linear', a=-1.0, b=1.0)

		self.complete_data_load()

	def load_EEGEyeNet(self):
		if self.target_dataset == 'DDir':
			self.target_X, self.target_y, self.target_train_list, self.target_val_list, self.target_test_list, self.target_train_list_full = data_reader_wrapper(self.target_dataset, self.perc_data_target)
		else: #WM
			self.target_X, self.target_y, self.target_train_list, self.target_val_list, self.target_test_list, self.target_train_list_full = data_loader_WM(self.perc_data_target)
		
		if self.type_ == 'r' or self.type_ == 'r_weighted':
			self.target_y = np.expand_dims(self.target_y[:, 0], axis=1)
			# print(np.min(self.target_y), np.max(self.target_y))
		
		elif self.type_ == 'theta' or self.type_ == 'theta_weighted':
			self.target_y = np.expand_dims(self.target_y[:, 1], axis=1)

	def load_WM(self):
		self.source_X, self.source_y, self.source_train_list, self.source_val_list, self.source_test_list, self.source_train_list_full = data_reader_wrapper('PS', self.perc_data_target)
		# self.source_X, self.source_y, self.source_train_list, self.source_test_list = data_loader_WM()
		# self.source_X, self.source_y, self.source_train_list, self.source_test_list = read_data_npz()
		
		if self.type_ == 'r' or self.type_ == 'r_weighted':
			self.source_y = np.expand_dims(self.source_y[:, 0], axis=1)
			# print(np.min(self.source_y), np.max(self.source_y))
		
		elif self.type_ == 'theta' or self.type_ == 'theta_weighted':
			self.source_y = np.expand_dims(self.source_y[:, 1], axis=1)


	def inv_scale(self, y):
		return self.scale_obj.inv_scale(y)

	def complete_data_load(self):
		#load the datasets
		self.load_EEGEyeNet()
		self.load_WM()
		#now scale the data
		self.source_scale_obj_eeg.set_params(self.source_X[self.source_train_list])
		self.source_X = self.source_scale_obj_eeg.rescale(self.source_X)

		self.target_scale_obj_eeg.set_params(self.target_X[self.target_train_list])
		self.target_X = self.target_scale_obj_eeg.rescale(self.target_X)

		#scale y
		all_y = np.concatenate([self.source_y, self.target_y], axis=0)
		#print(all_y.shape)
		self.scale_obj.set_params(all_y)
		del all_y
		self.source_y = self.scale_obj.rescale(self.source_y)
		self.target_y = self.scale_obj.rescale(self.target_y)

	def get_data_source(self):
		train_X = self.source_X[self.source_train_list]
		train_y = self.source_y[self.source_train_list]
		test_X = self.source_X[self.source_test_list]
		test_y = self.source_y[self.source_test_list]
		val_X = self.source_X[self.source_val_list]
		val_y = self.source_y[self.source_val_list]

		p_y_gmm = GaussianMixture(n_components=14, random_state=0).fit(self.source_y[self.source_train_list_full])	 # from 0 onwards to make the exponential work
		

		print(train_y.shape, val_y.shape, test_y.shape)
		# print(np.min(train_y), np.max(train_y), np.min(test_y), np.max(test_y))
		del self.source_X
		del self.source_y
		return train_X, train_y, test_X, test_y, val_X, val_y, p_y_gmm

	def get_data_target(self):
		train_X = self.target_X[self.target_train_list]
		train_y = self.target_y[self.target_train_list]
		test_X = self.target_X[self.target_test_list]
		test_y = self.target_y[self.target_test_list]
		val_X = self.target_X[self.target_val_list]
		val_y = self.target_y[self.target_val_list]

		if self.target_dataset == 'DDir':
			p_y_gmm = GaussianMixture(n_components=14, random_state=0).fit(self.target_y[self.target_train_list_full])
		else: #WM
			p_y_gmm = EM_Exp_GMM_1d(2, 2, self.target_y[self.target_train_list_full]+1)	 # from 0 onwards to make the exponential work
			p_y_gmm.run_EM()
		
		print(train_y.shape, val_y.shape, test_y.shape)
		# print(np.min(train_y), np.max(train_y), np.min(test_y), np.max(test_y))
		del self.target_X
		del self.target_y
		return  train_X, train_y, test_X, test_y, val_X, val_y, p_y_gmm


# obj = integrated_data(target_dataset='WM')
# source_train_X, source_train_y, source_test_X, source_test_y, source_val_X, source_val_y = obj.get_data_source()
# target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y = obj.get_data_target()
# print(source_train_X.shape, source_train_y.shape, source_test_X.shape, source_test_y.shape, target_train_X.shape, target_train_y.shape, target_test_X.shape, target_test_y.shape)
# print(np.min(source_train_y), np.max(source_train_y), np.min(source_test_y), np.max(source_test_y), np.min(target_train_y), np.max(target_train_y), np.min(target_test_y), np.max(target_test_y))