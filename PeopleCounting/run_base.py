from read_data import *
from train_vit import *
from plotter import *


# for training and testing id and ood
def run_pretrain(m_saveloc, r_saveloc, pretrain_loc=None, type_='r', target_dataset='DDir', perc_data=1.0, source_train=False, model='EEGNET-LSTM', seed=0):
    n_epochs = 100
    data_loader = integrated_data(type_, target_dataset, perc_data_target=perc_data)

    if pretrain_loc is None:
        if source_train == True:
            source_train_X, source_train_y, _, _, target_val_X, target_val_y = data_loader.get_data_source()
        else:
            source_train_X, source_train_y, _, _, target_val_X, target_val_y, _ = data_loader.get_data_target()
    else:
        source_train_X, source_train_y, _, _, target_val_X, target_val_y, _ = data_loader.get_data_target()
    
    #last setting of seed for the model training
    np.random.seed(seed)
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

def run_test_pretrain_paper(m_saveloc, r_saveloc, type_='r', in_='pix', min_1=None, max_1=None, min_2=None, max_2=None, target_dataset='DDir', source_plot_title="", target_plot_title=""):
    
    data_loader = integrated_data(type_, target_dataset)
    # source_train_X, source_train_y, source_test_X, source_test_y, _, _ = data_loader.get_data_source()
    target_train_X, target_train_y, target_test_X, target_test_y, target_val_X, target_val_y, _ = data_loader.get_data_target()

    # target_train_X, target_train_y, target_test_X, target_test_y = data_loader.get_data_source()
    # source_train_X, source_train_y, source_test_X, source_test_y = data_loader.get_data_target()

    enet_lstm_obj = LSTM_EEGNet(type_=type_)
    enet_lstm_obj.load_model(m_saveloc)
    ####################################################source################################################################################

    eye_train = data_loader.inv_scale(target_train_y)
    eye_test = data_loader.inv_scale(target_test_y)
    eye_val = data_loader.inv_scale(target_val_y)

    pred_train_y = data_loader.inv_scale(enet_lstm_obj.predict(target_train_X))
    pred_test_y = data_loader.inv_scale(enet_lstm_obj.predict(target_test_X))
    pred_val_y = data_loader.inv_scale(enet_lstm_obj.predict(target_val_X))

    # print(eye_train.shape, pred_train_y.shape)
    # # return
    # #x-predictions
    
    
    plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= target_plot_title, addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)

    ###################################################target######################################################################################
    # eye_train = data_loader.inv_scale(source_train_y)
    # eye_test = data_loader.inv_scale(source_test_y)
    # pred_train_y = data_loader.inv_scale(enet_lstm_obj.predict(source_train_X))
    # pred_test_y = data_loader.inv_scale(enet_lstm_obj.predict(source_test_X))
    
    # plot_predictions_paper(pred_test_y[:, 0], eye_test[:, 0], r_saveloc+"test_r_source_paper.png", title_= source_plot_title, addon="del {} (in {})".format(type_, in_), _min_use=min_2, _max_use=max_2)