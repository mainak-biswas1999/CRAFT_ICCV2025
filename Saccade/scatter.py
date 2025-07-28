from loader import *
from read_EEGEyenet import *
from LSTM_EEGNet import *
from pyramidal_cnn import *
from read_data import *
from read_deepak_data import *
from get_naive_baselines import *
import matplotlib.gridspec as gridspec
import pingouin as png
import sys
from plot_paper import *
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

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

def get_rmse(y, pred_y):
    #y = scale_obj.inv_scale(y)
    #pred_y = scale_obj.inv_scale(pred_y)
    
    #return np.mean(np.sqrt((y[:, 0] - pred_y[:, 0])**2 + (y[:, 1] - pred_y[:, 1])**2))
    return np.linalg.norm(y - pred_y, axis=1).mean()

def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y.ravel(), y_pred.ravel()))
    
def get_metric(y, pred_y, scale_obj):
    
    y = scale_obj.inv_scale(y)
    pred_y = scale_obj.inv_scale(pred_y)
    
    #print(y.shape, pred_y.shape)
    corr = png.corr(y, pred_y, method='percbend')
    r = corr['r'][0]
    if np.isnan(r):
        r = 0
    p = corr['p-val'][0]
    
    
    rmse = np.sqrt(np.mean((y - pred_y)**2))
    mae = np.mean(np.abs((y - pred_y)))
    
    return r, p, rmse, mae 

def plot_predictions_paper_wo_density(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
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

    # plt.scatter(y_age, pred_y_age, c=density2, cmap="hot", alpha=0.75, s=400, edgecolor='black', linewidth=3) #color='k', edgecolor='black')
    # plt.scatter(y_age, pred_y_age, alpha=0.9, s=400, edgecolor='black', linewidth=2, color='w', marker="^")
    plt.scatter(y_age, pred_y_age, alpha=0.9, s=400, edgecolor='black', linewidth=2, color='w', marker="o")
    
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
        # ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}, mae={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error, mae_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
        ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
        ################################print best fit#############################################
        A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
        w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
            
        y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
        # plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='plum')
        plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='indianred')
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

def display_scatter(m_saveloc, r_saveloc, type_='r', in_='pix', model='EEGNET-LSTM', target_dataset='DDir'):
    
    data_loader = integrated_data('r', target_dataset, perc_data_target=1.0)
    target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, p_y_gmm = data_loader.get_data_target()
    
    obj_cdist = LSTM_EEGNet()
    obj_cdist.load_model(m_saveloc+"Efects_Decoder_source/")
    ####################################################target################################################################################

    # eye_train = data_loader.inv_scale(target_train_y)
    eye_test = data_loader.inv_scale(target_test_y)
    # eye_val = data_loader.inv_scale(target_val_y)
    # pred_train_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_train_X))
    pred_test_y = data_loader.inv_scale(obj_cdist.predict(target_test_X))
    # pred_val_y = data_loader.inv_scale(obj_cdist.model_source.predict(target_val_X))

    plot_predictions_paper_wo_density(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper_noDensity.png", title_= "EEGNET-LSTM: CRAFT", addon="$|\\Delta\\vec{{r}}|$ (in {})".format(in_))


if __name__ == "__main__":
    import gc
    display_scatter("./Models/rebuttal_final/craft_perc_sup_sacc/20/1/", "./Results/rebuttal_final/craft_perc_sup_sacc/20/1/")
    