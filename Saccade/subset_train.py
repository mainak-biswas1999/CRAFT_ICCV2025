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
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]) # Notice here
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


def plot_predictions(pred_y, y, scale_obj, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
    import pingouin as png
    
    y_age = y = np.squeeze(y)
    pred_y_age = pred_y = np.squeeze(pred_y)
    
    #y_age = scale_obj.inv_scale(y)
    #pred_y_age = scale_obj.inv_scale(pred_y)
    
    _min = np.min([np.min(y_age), np.min(pred_y_age)]) 
    _max = np.max([np.max(y_age), np.max(pred_y_age)])
    
    #generate the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.scatter(y_age, pred_y_age, alpha=0.95, s=300, edgecolor='black', linewidth=1)
    plt.xlabel("Actual "+addon, fontsize=45)
    plt.ylabel("Predicted "+addon, fontsize=45)
    
    if _min_use is None and _max_use is None:
        plt.xlim(_min - 1, _max + 1)
        plt.ylim(_min - 1, _max + 1)
    elif _min_use is not None and _max_use is None:
        plt.xlim(_min_use, _max + 1)
        plt.ylim(_min_use, _max + 1)
    elif _min_use is None and _max_use is not None:
        plt.xlim(_min - 1, _max_use)
        plt.ylim(_min - 1, _max_use)
    else:
        plt.xlim(_min_use, _max_use)
        plt.ylim(_min_use, _max_use)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=28)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    ax.tick_params(width=8)
    plt.title(title_, fontsize=35, fontname="Myriad Pro")
    yex = np.linspace(_min - 1, _max + 1, 10000)
    if circ == False:
        corr = png.corr(np.squeeze(y_age), np.squeeze(pred_y_age), method='percbend')
        mse_error = np.round(np.sqrt(np.mean((y_age - pred_y_age)**2)), 2)
        mae_error = np.round(np.mean(np.abs((y_age - pred_y_age))), 2)
        #print(mse_error, mae_error)
        ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}, mae={}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), str(mse_error), str(mae_error)), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=28)
        ################################print best fit#############################################
        A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
        w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
            
        y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
        plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=3, color='red')
    else:
        corr, pval = png.circ_corrcc(np.squeeze(y_age), np.squeeze(pred_y_age), correction_uniform=True)
        mse_error = get_angle(np.squeeze(y_age), np.squeeze(pred_y_age))
        #print(mse_error, mae_error)
        ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}".format(str(np.round(np.abs(corr), 3)), np.maximum(np.round(pval, 3), 0.001), str(np.round(mse_error, 2))), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=28)
    
    ################################print y=x##################################################
    plt.plot(yex, yex, linestyle = 'dashed', linewidth=4, color='black')
    
    	
    #plt.title("r= {}, p= {}".format(np.round(corr['r'][0], 2), np.round(corr['p-val'][0], 3)))
    plt.savefig(saveloc)
    plt.tight_layout()
    plt.close()

  
def plot_time_series_result(train_t_r, train_t_p, test_t_r, test_t_p, _xlabel_, _ylabel_, _title_, img_saveloc, to_plot_p=False):
    fig, ax = plt.subplots(figsize=(14, 12))
    
    lines = [500, 700, 1370, 1970]
    t = 11.71 + 7.8125*np.arange(len(train_t_r))
    #lines = [1000, 1400]
    #t = 12.5 + 8.33*np.arange(len(train_t_r))
    #t = 10 + 20*np.arange(len(train_t_r))
    
    _min = np.min([np.min(test_t_r), np.min(train_t_r)]) 
    _max = np.max([np.max(train_t_r), np.max(test_t_r)])
    
    plt.plot(t, train_t_r, linewidth=4, label='Train', color='firebrick')
    plt.plot(t, test_t_r, linewidth=4, label='Test', color='royalblue')

    plt.xlabel(_xlabel_, size=25)
    plt.ylabel(_ylabel_, size=25)
    plt.title(_title_, size=30)
    
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    ax.tick_params(width=8)
    
    plt.ylim(_min - 0.05*np.abs(_min), _max + 0.05*np.abs(_max))
    
    
    for i in range(int(len(lines))):
        plt.axvline(x=lines[i], color='k', linewidth=2, linestyle='--')
    
    
    mid_pt = 0.45*(_min + _max)
    
    #plt.text(10, mid_pt, "Fixation", size=20, rotation='vertical', color='k', fontweight='bold')
    #plt.text(1020, mid_pt, "Cue", size=20, rotation='vertical', color='k', fontweight='bold')
    #plt.text(1440, mid_pt, "Stimulus (x16)", size=20, rotation='vertical', color='k', fontweight='bold')
    
    #plt.text(1400, mid_pt, "Saccade Onset", size=20, rotation='vertical', color='k', fontweight='bold')
    
    #plt.text(2000, mid_pt, "Probe Onset", size=20, rotation='vertical', color='k', fontweight='bold')
    
    plt.text(20, mid_pt, "Fixation", size=20, rotation='vertical', color='k', fontweight='bold')
    plt.text(520, mid_pt, "Stimulus", size=20, rotation='vertical', color='k', fontweight='bold')
    
    plt.text(1400, mid_pt, "Saccade Onset", size=20, rotation='vertical', color='k', fontweight='bold')
    
    plt.text(2000, mid_pt, "Probe Onset", size=20, rotation='vertical', color='k', fontweight='bold')
    plt.legend(fontsize=25, loc ="lower left")
    
    
    if to_plot_p == True:
        #sig_places_train = 10 + 20*np.where(train_t_p < 0.05)[0]
        sig_places_train = 11.71 + 7.8125*np.where(train_t_p < 0.05)[0]
        #sig_places_test = 10 + 20*np.where(test_t_p < 0.05)[0]
        sig_places_test = 11.71 + 7.8125*np.where(test_t_p < 0.05)[0]
        plt.scatter(sig_places_train, np.ones(sig_places_train.shape)*_max*0.9, marker='*', c='firebrick', linewidth=2)
        plt.scatter(sig_places_test, np.ones(sig_places_test.shape)*_max*0.8, marker='*', c='royalblue', linewidth=2)
    
    
    plt.tight_layout()
    plt.savefig(img_saveloc)
    plt.close()
    
def get_accuracy(y, pred_y):
    y = np.squeeze(y)
    pred_y = np.squeeze(pred_y)
    return np.round(np.sum(y == (pred_y>0.5))*1./len(y), 3)

def get_rmse(y, pred_y):
    #y = scale_obj.inv_scale(y)
    #pred_y = scale_obj.inv_scale(pred_y)
    
    #return np.mean(np.sqrt((y[:, 0] - pred_y[:, 0])**2 + (y[:, 1] - pred_y[:, 1])**2))
    return np.linalg.norm(y - pred_y, axis=1).mean()

def get_angle(y, y_pred):
    return np.sqrt(np.mean(np.square(np.arctan2(np.sin(y.ravel() - y_pred.ravel()), np.cos(y.ravel() - y_pred.ravel())))))

def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y.ravel(), y_pred.ravel()))

def get_radius(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    
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

def run_pretrain(m_saveloc, r_saveloc, pretrain_loc=None, type_='r', target_dataset='DDir', perc_data=1.0, source_train=False, model='EEGNET-LSTM', seed=0, perc_label=0.5):
    n_epochs = 100
    data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)

    if pretrain_loc is None:
        if source_train == True:
            source_train_X, source_train_y, _, _, target_val_X, target_val_y, _ = data_loader.get_data_source()
        else:
            source_train_X, source_train_y, _, _, target_val_X, target_val_y, _ = data_loader.get_data_target()
    else:
        source_train_X, source_train_y, _, _, target_val_X, target_val_y, _ = data_loader.get_data_target()
    
    #last setting of seed for the model training
    np.random.seed(seed)
    source_train_X, source_train_y, _, _ = return_labeled_subets(source_train_X, source_train_y, perc_label)

    if model == 'EEGNET-LSTM':
        enet_lstm_obj = LSTM_EEGNet(type_=type_)
    else:
        enet_lstm_obj = SpyrCNN_wrapper(type_=type_)
    if pretrain_loc is None:
        enet_lstm_obj.make_model()
    else:
        enet_lstm_obj.load_model(pretrain_loc)
    enet_lstm_obj.train_model(source_train_X, source_train_y, m_saveloc, r_saveloc, num_epochs=n_epochs, val_data=target_val_X, val_label=target_val_y)
    
    del source_train_X, source_train_y, target_val_X, target_val_y
    del enet_lstm_obj

def run_test_pretrain(m_saveloc, r_saveloc, type_='r', in_='pix', min_1=None, max_1=None, min_2=None, max_2=None, target_dataset='DDir', source_plot_title="", target_plot_title="", perc_data=1.0, source_test=True, target_test=True, model='EEGNET-LSTM'):
    

    data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)
    
    if model == 'EEGNET-LSTM':
        enet_lstm_obj = LSTM_EEGNet(type_=type_)
    else:
        enet_lstm_obj = SpyrCNN_wrapper(type_=type_)
    enet_lstm_obj.load_model(m_saveloc)
    fptr = open(r_saveloc+"performance_predictions.txt", 'w')
    
    ####################################################target################################################################################
    if target_test == True:
        target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_target()
        eye_train = data_loader.inv_scale(target_train_y)
        eye_test = data_loader.inv_scale(target_test_y)
        eye_val = data_loader.inv_scale(target_val_y)

        pred_train_y = data_loader.inv_scale(enet_lstm_obj.predict(target_train_X))
        pred_test_y = data_loader.inv_scale(enet_lstm_obj.predict(target_test_X))
        pred_val_y = data_loader.inv_scale(enet_lstm_obj.predict(target_val_X))

        print(eye_train.shape, pred_train_y.shape)
        # return
        #x-predictions
        
        if type_ == 'r':
            plot_predictions(pred_train_y[:, 0], eye_train[:, 0], None, r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
            plot_predictions(pred_test_y[:, 0], eye_test[:, 0], None, r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1)
            plot_predictions(pred_val_y[:, 0], eye_val[:, 0], None, r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1) 
            
            plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= target_plot_title, addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
            
            print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y), get_rmse1(eye_val, pred_val_y)), file=fptr)
        elif type_ == 'theta':
            plot_predictions(pred_train_y[:, 0], eye_train[:, 0], None, r_saveloc+"train_r_target.png", title_="Train (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
            plot_predictions(pred_test_y[:, 0], eye_test[:, 0], None, r_saveloc+"test_r_target.png", title_="Test (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
            plot_predictions(pred_val_y[:, 0], eye_val[:, 0], None, r_saveloc+"val_r_target.png", title_="Val (Target): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_1, _max_use=max_1, circ=True)
            print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_angle(eye_train, pred_train_y), get_angle(eye_test, pred_test_y),  get_angle(eye_val, pred_val_y)), file=fptr)

        del target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y
    ###################################################source######################################################################################
    if source_test == True:
        source_train_X, source_train_y, source_test_X, source_test_y, _, _, _ = data_loader.get_data_source()
        eye_train = data_loader.inv_scale(source_train_y)
        eye_test = data_loader.inv_scale(source_test_y)
        pred_train_y = data_loader.inv_scale(enet_lstm_obj.predict(source_train_X))
        pred_test_y = data_loader.inv_scale(enet_lstm_obj.predict(source_test_X))
        #x-predictions
        if type_ == 'r' or type_ == 'r_weighted':
            plot_predictions(pred_train_y[:, 0], eye_train[:, 0], None, r_saveloc+"train_r_source.png", title_="Train (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
            plot_predictions(pred_test_y[:, 0], eye_test[:, 0], None, r_saveloc+"test_r_source.png", title_="Test  (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)
            
            plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_source_paper.png", title_=source_plot_title, addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)


            print("Train RMSE (Target): {}  \nTest RMSE (Target): {}".format(get_rmse1(eye_train, pred_train_y), get_rmse1(eye_test, pred_test_y)), file=fptr)
        elif type_ == 'theta':
            plot_predictions(pred_train_y[:, 0], eye_train[:, 0], None, r_saveloc+"train_r_source.png", title_="Train (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2, circ=True)
            plot_predictions(pred_test_y[:, 0], eye_test[:, 0], None, r_saveloc+"test_r_source.png", title_="Test  (Source): {}".format(type_), addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2, circ=True)
            print("Train RMSE (source): {}  \nTest RMSE (source): {}".format(get_angle(eye_train, pred_train_y), get_angle(eye_test, pred_test_y)), file=fptr)
        
        del source_train_X, source_train_y, source_test_X, source_test_y

    del enet_lstm_obj
    fptr.close()


if __name__ == "__main__":
    import gc
    from multiprocessing import Process

    base_mloc = "./Models/rebuttal/perc_sup_sacc/"
    base_rloc = "./Results/rebuttal/perc_sup_sacc/"
    for i in [1, 2, 5, 10, 25, 50, 75]:
        if not os.path.exists(base_rloc+str(i)):
                os.mkdir(base_rloc+str(i))
        if not os.path.exists(base_mloc+str(i)):
                os.mkdir(base_mloc+str(i))
        for j in range(1,2):
            if not os.path.exists(base_rloc+str(i)+"/"+str(j)):
                os.mkdir(base_rloc+str(i)+"/"+str(j))
            if not os.path.exists(base_mloc+str(i)+"/"+str(j)):
                os.mkdir(base_mloc+str(i)+"/"+str(j))
            
            run_pretrain(base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", pretrain_loc="./Models/iclr/EEGNET-LSTM/source_VS/", target_dataset='DDir', type_='r', perc_label=(0.01*i), seed=j, model='EEGNET-LSTM')
            run_test_pretrain(base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", type_='r', in_='pix', target_dataset='DDir', source_test=False, target_test=True, model='EEGNET-LSTM')
            gc.collect()
            tf.keras.backend.clear_session()