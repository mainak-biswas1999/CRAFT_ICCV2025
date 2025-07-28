from read_EEGEyenet import *
from read_deepak_data import *
from eye_dat_prep import *

def get_rmse(y, pred_y):
    #y = scale_obj.inv_scale(y)
    #pred_y = scale_obj.inv_scale(pred_y)
    
    #return np.mean(np.sqrt((y[:, 0] - pred_y[:, 0])**2 + (y[:, 1] - pred_y[:, 1])**2))
    return np.linalg.norm(y - pred_y, axis=1).mean()

def get_radius(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    
def get_angle(y, y_pred):
    return np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - y_pred.ravel()), np.cos(y - y_pred.ravel())))))

def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y, y_pred.ravel()))
    
def find_naivebaseline(dataset='DDir', addon='ddir_target'):
    _, eye_data, train_split, val_split, test_split = data_reader_wrapper(dataset)
    
    eye_train = eye_data[train_split]
    eye_test = eye_data[test_split]
    eye_val = eye_data[val_split]
    
    mean_train = np.mean(eye_train, axis=0, keepdims=True)
    
    pred_train_y = np.repeat(mean_train, eye_train.shape[0], axis=0)
    pred_test_y = np.repeat(mean_train, eye_test.shape[0], axis=0)
    pred_val_y = np.repeat(mean_train, eye_val.shape[0], axis=0)
    
    print(mean_train, mean_train.shape, pred_train_y.shape, eye_train.shape, pred_test_y.shape, eye_test.shape, pred_val_y.shape, eye_val.shape)
    
    
    
    fptr = open("./Results/ps_eegeyeNet/stats/performance_naive_{}.txt".format(addon), 'w')
    print("Train RMSE: {} \nValidation RMSE: {} \nTest RMSE: {}".format(get_rmse1(eye_train[:, 0], pred_train_y[:, 0]), get_rmse1(eye_val[:, 0], pred_val_y[:, 0]), get_rmse1(eye_test[:, 0], pred_test_y[:, 0])), file=fptr)
    print("Train RMSE: {} \nValidation RMSE: {} \nTest RMSE: {}".format(get_angle(eye_train[:, 1], pred_train_y[:, 1]), get_angle(eye_val[:, 1], pred_val_y[:, 1]), get_angle(eye_test[:, 1], pred_test_y[:, 1])), file=fptr)
    fptr.close()
    
    # fptr = open("./Results/EEGNet_LSTM_EEGEYENet/DPos/performance_naive.txt", 'w')
    # print("Train RMSE: {} \nValidation RMSE: {} \nTest RMSE: {}".format(get_rmse(eye_train, pred_train_y), get_rmse(eye_val, pred_val_y), get_rmse(eye_test, pred_test_y)), file=fptr)
    # fptr.close()
    # print(eye_mean.shape, np.round(eye_mean, 4))


def find_naivebaseline_WM():
    # _, eye_data, train_list, test_list = data_loader_WM()
    _, eye_data, train_list, test_list = read_data_npz()

    eye_train = eye_data[train_list]
    eye_test = eye_data[test_list]
    
    mean_train = np.mean(eye_train, axis=0, keepdims=True)
    
    pred_train_y = np.repeat(mean_train, eye_train.shape[0], axis=0)
    pred_test_y = np.repeat(mean_train, eye_test.shape[0], axis=0)
    
    # print(np.min(eye_train), np.max(eye_train), np.mean(eye_train))
    
    #fptr = open("./Results/EEGNet_LSTM_EEGEYENet/DDir/performance_naive.txt", 'w')
    #print("Train RMSE: {} \nValidation RMSE: {} \nTest RMSE: {}".format(get_rmse1(eye_train[:, 0], pred_train_y[:, 0]), get_rmse1(eye_val[:, 0], pred_val_y[:, 0]), get_rmse1(eye_test[:, 0], pred_test_y[:, 0])), file=fptr)
    #print("Train RMSE: {} \nValidation RMSE: {} \nTest RMSE: {}".format(get_angle(eye_train[:, 1], pred_train_y[:, 1]), get_angle(eye_val[:, 1], pred_val_y[:, 1]), get_angle(eye_test[:, 1], pred_test_y[:, 1])), file=fptr)
    #fptr.close()
    
    # fptr = open("./Results/EEGNet_LSTM_WM/DPos/performance_naive.txt", 'w')
    # fptr = open("./Results/EEGNet_LSTM_WM_DDir/single_model/performance_naive_g50.txt", 'w')
    fptr = open("./Results/eeg_gaze_robots/stats/performance_naive.txt", 'w')
    print("Radius ---- Train RMSE: {} \nTest RMSE: {}".format(get_rmse1(eye_train[:, 0], pred_train_y[:, 0]), get_rmse1(eye_test[:, 0], pred_test_y[:, 0])), file=fptr)
    print("Theta ---- Train RMSE: {} \nTest RMSE: {}".format(get_angle(eye_train[:, 1], pred_train_y[:, 1]), get_angle(eye_test[:, 1], pred_test_y[:, 1])), file=fptr)

    fptr.close()
    #print(eye_mean.shape, np.round(eye_mean, 4))
    
# find_naivebaseline() 
# find_naivebaseline(dataset='PS', addon='ps_source') 
