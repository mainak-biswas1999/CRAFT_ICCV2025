import pingouin as png
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import auc

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

def plot_curves2(loss1, loss2, __title__, y_label, n_epochs, saveloc, x_label='Epoch', legends=[], x_axis_vals=[]):
    plt.figure(figsize=(12, 8))
    if len(x_axis_vals) != 0:
        plt.plot(x_axis_vals, loss1)
        plt.plot(x_axis_vals, loss2)
    else:
        plt.plot(np.linspace(0, n_epochs, len(loss1)), loss1)
        plt.plot(np.linspace(0, n_epochs, len(loss1)), loss2)
    
    plt.xlabel(x_label, size=35)
    plt.ylabel(y_label, size=35)
    plt.legend(legends, fontsize='25', loc='lower right')
    plt.title(__title__, size=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.tick_params(width=8)
    plt.tight_layout()
    plt.savefig(saveloc)
    plt.close()

def plot_curves(loss, __title__, y_label, n_epochs, saveloc, x_label='Epoch', x_axis_vals=[]):
    plt.figure(figsize=(12, 8))
    if len(x_axis_vals) != 0:
        plt.plot(x_axis_vals, loss)
    else:
        plt.plot(np.linspace(0, n_epochs, len(loss)), loss)
    plt.xlabel(x_label, size=35)
    plt.ylabel(y_label, size=35)
    plt.title(__title__, size=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.tick_params(width=8)
    plt.tight_layout()
    plt.savefig(saveloc)
    plt.close()

def plot_predictions(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
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
        ax.text(0.02, 0.88, "r = {}, p < {} \nrmse={}, mae={}".format(np.round(corr['r'][0], 3), np.maximum(np.round(corr['p-val'][0], 3), 0.001), str(mse_error), str(mae_error)), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=28)
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

def print_performance(saveloc):
    perc_data = [0.2, 0.4, 0.6]
    naive_rmse = np.array([[7.92, 8.01, 8.15], [7.91, 7.96, 8.00], [7.94, 7.93, 7.93]])

    tl_rmse = np.array([[7.69, 7.66, 6.89], [6.88, 7.03, 6.51], [6.52, 7.03, 6.63]])
    tl_r = np.array([[0.37, 0.29, 0.58], [0.52, 0.50, 0.66], [0.60, 0.50, 0.59]])

    tasfar_rmse = np.array([[7.63, 7.68, 7.11], [6.97, 7.09, 6.67], [6.67, 7.02, 6.67]])
    tasfar_r = np.array([[0.37, 0.30, 0.58], [0.54, 0.50, 0.68], [0.60, 0.53, 0.58]])

    datafree_rmse = np.array([[7.48, 7.57, 7.02], [6.94, 6.96, 6.64], [6.50, 7.01, 6.80]])
    datafree_r = np.array([[0.44, 0.48, 0.57], [0.52, 0.52, 0.62], [0.58, 0.47, 0.59]])

    craft_rmse = np.array([[7.16, 7.36, 6.89], [6.56, 6.64, 6.60], [6.60, 6.80, 6.45]])
    craft_r = np.array([[0.49, 0.46, 0.57], [0.63, 0.64, 0.60], [0.61, 0.56, 0.63]])

    bbcn_rmse = np.array([[7.69, 8.33, 7.98], [7.46, 7.58, 7.65], [7.60, 7.65, 7.28]])
    bbcn_r = np.array([[0.37, 0.22, 0.25], [0.35, 0.37, 0.34], [0.41, 0.28, 0.43]])

    mixup_rmse = np.array([[7.46, 7.62, 8.04], [7.02, 7.45, 7.30], [6.50, 6.84, 7.23]])
    mixup_r = np.array([[0.43, 0.26, 0.33], [0.51, 0.39, 0.45], [0.61, 0.55, 0.49]])

    fptr = open(saveloc+"results_summary_3.txt", 'w')
    for i in range(len(perc_data)):
        print("Naive", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(naive_rmse[i]), 2), np.round(np.std(naive_rmse[i])/np.sqrt(len(naive_rmse[i])), 2)), file=fptr)
        print("TL", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tl_rmse[i]), 2), np.round(np.std(tl_rmse[i])/np.sqrt(len(tl_rmse[i])), 2), np.round(np.mean(tl_r[i]), 2), np.round(np.std(tl_r[i])/np.sqrt(len(tl_r[i])), 2)), file=fptr)
        print("TASFAR", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tasfar_rmse[i]), 2), np.round(np.std(tasfar_rmse[i])/np.sqrt(len(tasfar_rmse[i])), 2), np.round(np.mean(tasfar_r[i]), 2), np.round(np.std(tasfar_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
        print("DataFree", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(datafree_rmse[i]), 2), np.round(np.std(datafree_rmse[i])/np.sqrt(len(datafree_rmse[i])), 2), np.round(np.mean(datafree_r[i]), 2), np.round(np.std(datafree_r[i])/np.sqrt(len(datafree_r[i])), 2)), file=fptr) 
        print("CRAFT", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(craft_rmse[i]), 2), np.round(np.std(craft_rmse[i])/np.sqrt(len(craft_rmse[i])), 2), np.round(np.mean(craft_r[i]), 2), np.round(np.std(craft_r[i])/np.sqrt(len(craft_r[i])), 2)), file=fptr)         
        print("BBCN", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(bbcn_rmse[i]), 2), np.round(np.std(bbcn_rmse[i])/np.sqrt(len(bbcn_rmse[i])), 2), np.round(np.mean(bbcn_r[i]), 2), np.round(np.std(bbcn_r[i])/np.sqrt(len(bbcn_r[i])), 2)), file=fptr)   
        print("Progressive Mixup: ", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(mixup_rmse[i]), 2), np.round(np.std(mixup_rmse[i])/np.sqrt(len(mixup_rmse[i])), 2), np.round(np.mean(mixup_r[i]), 2), np.round(np.std(mixup_r[i])/np.sqrt(len(mixup_r[i])), 2)), file=fptr)
        print("-------------------------------------------------------------------------", file=fptr)
    fptr.close()

def validation(saveloc):
    # validation -- eeg net
    

    #previous experiments:

    #eye-decode
    alpha = [0.01, 0.1, 1.0]

    tasfar = np.array([[102.12, 98.98, 96.63], [96.91, 97.92, 94.20], [97.93, 98.72, 94.74]]) 
    datafree = np.array([[107.54, 104.60, 96.23], [105.61, 103.37, 93.27], [106.43, 106.87, 93.77]])
    # craft = np.array([[106.65, 104.09, 98.25], [104.24, 103.01, 96.94], [115.62,  116.16, 120.71]])
    #age
    perc_data = [0.01, 0.10, 0.10]  #actually alpha
    tasfar_rmse = np.array([[7.68, 7.61, 6.99], [7.63, 7.68, 7.11], [7.75, 7.69, 6.96]]) 
    tasfar_r = np.array([[0.34, 0.32, 0.53], [0.37, 0.30, 0.58], [0.30, 0.38, 0.53]]) 

    datafree_rmse = np.array([[8.28, 7.77, 7.42], [7.60, 7.46, 7.00], [8.73, 8.56, 8.82]])
    datafree_r = np.array([[0.25, 0.27, 0.42], [0.49, 0.40, 0.62], [0.03, 0.05, 0.12]])

    craft_rmse = np.array([[7.54, 7.48, 6.90], [7.16, 7.36, 6.89], [8.45, 8.45, 9.01]])
    craft_r = np.array([[0.40, 0.36, 0.60], [0.49, 0.46, 0.57], [0.41, 0.29, 0.41]])
    fptr = open(saveloc+"alphas_summary.txt", 'w')
    for i in range(len(perc_data)):
        print("TASFAR", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(tasfar_rmse[i]), 2), np.round(np.std(tasfar_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(tasfar_r[i]), 2), np.round(np.std(tasfar_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)
        print("DataFree", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(datafree_rmse[i]), 2), np.round(np.std(datafree_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(datafree_r[i]), 2), np.round(np.std(datafree_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr) 
        print("CRAFT", file=fptr)
        print("perc data: {}, rmse= {} $\\pm$ {}, r= {} $\\pm$ {}".format(perc_data[i], np.round(np.mean(craft_rmse[i]), 2), np.round(np.std(craft_rmse[i])/np.sqrt(len(perc_data)), 2), np.round(np.mean(craft_r[i]), 2), np.round(np.std(craft_r[i])/np.sqrt(len(perc_data)), 2)), file=fptr)         
        print("-------------------------------------------------------------------------", file=fptr)
    fptr.close()


    # sup_time = np.array([, 9.76])
    # tasfar_time = np.array([, ])
    # datafree_time = np.array([, ])
    # craft_time = np.array([, ])

if __name__=="__main__":
    print_performance("./Results/rebuttal/")
    # validation("./Results/hyperparameters/")