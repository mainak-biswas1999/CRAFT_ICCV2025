import numpy as np
import matplotlib.pyplot as plt
import pingouin as png
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm

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
    
def plot_logit_histograms(xt, xp, saveloc, title_, xlabel_, l1='ID', l2='OOD', n_gmm=2):
    # plot the histogram
    xt = xt
    if len(xt.shape) == 1:
        xt = np.expand_dims(xt, axis=-1)
    if xp is not None:
        xp = xp
        if len(xp.shape) == 1:
            xp = np.log(np.expand_dims(xp, axis=-1)+2)

    gmm_xt = GaussianMixture(n_components=n_gmm, random_state=0).fit(xt)
    if xp is not None:
        gmm_xp = GaussianMixture(n_components=n_gmm, random_state=0).fit(xp)
        min_val_p = np.min(xp)
        max_val_p = np.max(xp)
        min_val_p =  min_val_p - 0.1*np.abs(min_val_p)
        max_val_p = max_val_p + 0.1 * np.abs(max_val_p)
    
    min_val_t = np.min(xt)
    max_val_t = np.max(xt)
    min_val_t =  min_val_t - 0.1*np.abs(min_val_t)
    max_val_t = max_val_t + 0.1 * np.abs(max_val_t)

    x_range_t = np.linspace(min_val_t, max_val_t, 10000)
    # the score samples returns the log-likelihood of the data
    density_xt = np.exp(gmm_xt.score_samples(np.expand_dims(x_range_t, axis=-1)))      #plot the density
    if xp is not None:
        x_range_p = np.linspace(min_val_p, max_val_p, 10000)
        density_xp = np.exp(gmm_xp.score_samples(np.expand_dims(x_range_p, axis=-1)))
        # print(density_xt.shape, density_xp.shape)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    plt.locator_params(axis='both', nbins=6) 
    ax.tick_params(width=8)
    
    plt.xlabel(xlabel_, fontsize=45)
    plt.ylabel("Density", fontsize=45)
    plt.title(title_, fontsize=55)
    
    # plt.plot(x, density_xt, alpha=1, color='royalblue', linewidth=5, label='Positive')
    # plt.plot(x_range_t, np.clip(density_xt, 0, 0.20), alpha=1, color='#f77e00', linewidth=5, label='Positive')
    plt.plot(x_range_t, density_xt, alpha=1, color='#f77e00', linewidth=5, label=l1)
    # plt.fill_between(x, density_xt, alpha=0.3, color='skyblue')
    # plt.fill_between(x_range_t, np.clip(density_xt, 0, 0.20), alpha=0.3, color='#fbd8b2')
    plt.fill_between(x_range_t, density_xt, alpha=0.3, color='#fbd8b2')
    if xp is not None:
        # plt.plot(x, density_xp, alpha=1, color='firebrick', linewidth=5, label='Negative')
        # plt.plot(x_range_p, np.clip(density_xp, 0, 0.20), alpha=1, color='#2e7dfe', linewidth=5, label='Negative')
        plt.plot(x_range_p, density_xp, alpha=1, color='#2e7dfe', linewidth=5, label=l2)
        # plt.fill_between(x, density_xp, alpha=0.3, color='lightcoral')
        # plt.fill_between(x_range_p, np.clip(density_xp, 0, 0.20), alpha=0.3, color='#b6d8fe')
        plt.fill_between(x_range_p, density_xp, alpha=0.3, color='#b6d8fe')

    # plt.axvline(x=0.0, color='k', linestyle='--', linewidth=5)
    

    plt.legend(fontsize=30)
    plt.savefig(saveloc, dpi=300, bbox_inches="tight")
    plt.close()

def plot_predictions_paper(pred_y, y, saveloc, title_="", addon= "", _min_use=None, _max_use=None, circ=False):
    y = np.squeeze(y)
    y_age = y # np.log(y+5)
    pred_y = np.squeeze(pred_y)
    pred_y_age = pred_y #np.log(pred_y + 5)

    #y_age = scale_obj.inv_scale(y)
    #pred_y_age = scale_obj.inv_scale(pred_y)
    
    _min = np.min([np.min(y_age), np.min(pred_y_age)]) 
    _max = np.max([np.max(y_age), np.max(pred_y_age)])
    
    if _min_use is None and _max_use is None:
        x_min_use, x_max_use = _min - 0.1, _max + 0.1
        y_min_use, y_max_use = _min - 0.1, _max + 0.1
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
        corr = png.corr(np.squeeze(y), np.squeeze(pred_y), method='percbend')
        mse_error = np.round(np.sqrt(np.mean((y - pred_y)**2)), 2)
        mae_error = np.round(np.mean(np.abs((y - pred_y))), 2)
        #print(mse_error, mae_error)
        ax.text(0.02, 0.88, "r = {:.2f}, p < {} \nrmse={:.2f}, mae={:.2f}".format(np.round(corr['r'][0], 2), np.maximum(np.round(corr['p-val'][0], 3), 0.001), mse_error, mae_error), horizontalalignment='left', fontname="Myriad Pro", verticalalignment='bottom', transform=ax.transAxes, fontsize=30)
        ################################print best fit#############################################
        A = np.append(np.ones((len(pred_y_age), 1)), np.expand_dims(y_age, axis=1), axis=1)
        w = np.linalg.inv(A.T@A) @ (A.T @ pred_y_age)
            
        y_tilde = (np.append(np.ones((len(yex), 1)), np.expand_dims(yex, axis=1), axis=1)) @ w
        plt.plot(yex, y_tilde, linestyle = 'dashed', linewidth=7, zorder=10, color='rebeccapurple')
    else:
        corr, pval = png.circ_corrcc(np.squeeze(y), np.squeeze(pred_y), correction_uniform=True)
        mse_error = get_angle(np.squeeze(y), np.squeeze(pred_y))
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
    #   cap.set_markeredgewidth(4)

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


if __name__ == '__main__':
    pass