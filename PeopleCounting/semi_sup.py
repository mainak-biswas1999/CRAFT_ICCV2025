from plotter import *
from read_data import *
from resnet import *


def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y.ravel(), y_pred.ravel()))

    
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

def naive_model(saveloc):
    _, y_train_id, _, y_val_id, _, _, y_train_ood, _, y_val_ood, _, y_test_ood, rescale_obj = load_id_ood_people()
    
    fptr = open(saveloc, 'w')
    y_train_ood = rescale_obj.inv_scale(y_train_ood)
    pred_ood = np.mean(y_train_ood)
    y_val_ood = rescale_obj.inv_scale(y_val_ood)
    y_test_ood = rescale_obj.inv_scale(y_test_ood)

    print("OOD Train loss: {} \nOOD Val loss: {} \nOOD Test loss: {}".format(mse_loss(y_train_ood, pred_ood), mse_loss(y_val_ood, pred_ood), mse_loss(y_test_ood, pred_ood)), file=fptr)

    y_train_id = rescale_obj.inv_scale(y_train_id)
    pred_id = np.mean(y_train_id)
    y_val_id = rescale_obj.inv_scale(y_val_id)

    print("ID Train loss: {} \nID Val loss: {}".format(mse_loss(y_train_id, pred_id), mse_loss(y_val_id, pred_id)), file=fptr)
    fptr.close()

def plot_distributions():
    X_train_id, y_train_id, X_val_id, y_val_id, X_test_id, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people()

    y_train_id = rescale_obj.inv_scale(y_train_id)
    y_val_id = rescale_obj.inv_scale(y_val_id)
    
    y_id = np.concatenate([y_train_id, y_val_id])

    y_train_ood = rescale_obj.inv_scale(y_train_ood)
    y_val_ood = rescale_obj.inv_scale(y_val_ood)
    y_test_ood = rescale_obj.inv_scale(y_test_ood)

    y_ood = np.concatenate([y_train_ood, y_val_ood, y_test_ood])
    plot_logit_histograms(y_id, y_ood, "./results/stats/label_dist_new.png", "Label Distribution (log scale)")


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


def return_labeled_subets_rnd(X, y, perc_label):
    n_lab = int(y.shape[0]*perc_label)
    indices = np.arange(y.shape[0])
    lab = np.random.choice(y.shape[0], n_lab, replace=False)
    ulab = np.delete(indices, lab)

    return X[lab], y[lab], X[ulab], y[ulab] 


def run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, m_saveloc, r_saveloc, pretrain_loc, seed=0, perc_label=0.5):
    np.random.seed(seed)
    X_train_ood, y_train_ood, _, _ = return_labeled_subets(X_train_ood, y_train_ood, perc_label)
    print(X_train_ood.shape)
    obj = resnet_model()
    obj.load_model(pretrain_loc)
    obj.train_model(X_train_ood, y_train_ood, m_saveloc, r_saveloc, num_epochs=10, val_data=X_val_ood, val_label=y_val_ood)
    
    del X_train_ood, y_train_ood
    del obj


def run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, m_saveloc, r_saveloc):
    obj = resnet_model()
    obj.load_model(m_saveloc)
    
    y_train = rescale_obj.inv_scale(y_train_ood)
    y_val = rescale_obj.inv_scale(y_val_ood)
    y_test = rescale_obj.inv_scale(y_test_ood)
    
    pred_train_y = rescale_obj.inv_scale(obj.predict(X_train_ood))
    pred_val_y = rescale_obj.inv_scale(obj.predict(X_val_ood))
    pred_test_y = rescale_obj.inv_scale(obj.predict(X_test_ood))
    
    plot_predictions_paper(pred_train_y[:, 0], y_train[:, 0], r_saveloc+"train_r_target_paper.png", title_='Train', addon=" # People") #, )
    plot_predictions_paper(pred_val_y[:, 0], y_val[:, 0], r_saveloc+"val_r_target_paper.png", title_= 'Validation', addon=" # People")
    plot_predictions_paper(pred_test_y[:, 0], y_test[:, 0], r_saveloc+"test_r_target_paper.png", title_= 'Attn-Resnet: TL', addon=" # People")

    del obj


def run_test_naive(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, r_saveloc, perc_label):
    
    _, target_train_y_lab, _, _ = return_labeled_subets(X_train_ood, y_train_ood, perc_label)
    fptr = open(r_saveloc+"performance_predictions.txt", 'w')

    y_train =  rescale_obj.inv_scale(y_train_ood)
    y_val =  rescale_obj.inv_scale(y_val_ood)
    y_test =  rescale_obj.inv_scale(y_test_ood)
    mean_pred = np.mean(rescale_obj.inv_scale(target_train_y_lab))

    pred_train_y = np.ones(y_train.shape)*mean_pred
    pred_val_y = np.ones(y_val.shape)*mean_pred
    pred_test_y = np.ones(y_test.shape)*mean_pred
    
    print("Train RMSE (Target): {}  \nTest RMSE (Target): {} \nVal RMSE (Target): {}".format(get_rmse1(y_train, pred_train_y), get_rmse1(y_test, pred_test_y), get_rmse1(y_val, pred_val_y)), file=fptr)
    fptr.close()
    


if __name__=='__main__':
    import gc
    
    base_mloc = "./models/naive/"
    base_rloc = "./results/naive/"
    _, _, _, _, _, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people(log_scale=False)
    for i in [2, 5, 10, 20]:
        if not os.path.exists(base_rloc+str(i)):
                os.mkdir(base_rloc+str(i))
        if not os.path.exists(base_mloc+str(i)):
                os.mkdir(base_mloc+str(i))
        for j in range(3):
            if not os.path.exists(base_rloc+str(i)+"/"+str(j)):
                os.mkdir(base_rloc+str(i)+"/"+str(j))
            if not os.path.exists(base_mloc+str(i)+"/"+str(j)):
                os.mkdir(base_mloc+str(i)+"/"+str(j))
            
            # run_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/", pretrain_loc="./models/nwpu_base_model/", seed=j, perc_label=(0.01*i))
            # run_test_finetune(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, base_mloc+str(i)+"/"+str(j)+"/", base_rloc+str(i)+"/"+str(j)+"/")
            run_test_naive(X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj, base_rloc+str(i)+"/"+str(j)+"/", perc_label=(0.01*i))
            gc.collect()
            tf.keras.backend.clear_session()