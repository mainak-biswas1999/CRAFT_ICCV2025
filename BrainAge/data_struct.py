import pickle
import numpy as np
import matplotlib.pyplot as plt


class Details:
    def __init__(self, scans=None, sex=None, cog_scores=None, conn_mat=None, t1_scan=None, diagnosis=None, sub_name=None, age=None):
        self.sub_name = sub_name
        self.scans = scans
        self.diagnosis = diagnosis
        self.sex = sex
        self.cog_scores = cog_scores
        self.conn_mat = conn_mat
        self.age = age
    
    def __str__(self):
        return f"sub_name={self.sub_name}\nscans={self.scans}\n" \
               f"diagnosis={self.diagnosis}\nsex={self.sex}\ncog_scores={self.cog_scores}\n" \
               f"conn_mat={self.conn_mat}" 



def normalize_single_age(a1, loc_to_save):
    mn, mx = np.min(a1), np.max(a1)
    np.save(loc_to_save, np.array([mn, mx]))
    a1_scaled = (a1 - mn)/(mx-mn)
    return a1_scaled

def normalize_single_age_given(a1, loc_to_save):
    mn_mx = np.load(loc_to_save)
    a1_scaled = (a1 - mn_mx[0])/(mn_mx[1] - mn_mx[0])
    return a1_scaled

def inv_prep(X, loc_to_save):
    mn_mx = np.load(loc_to_save)
    feat_mg  = X*(mn_mx[1] - mn_mx[0]) + mn_mx[0] 
    return feat_mg


class TLSA_data:
    def __init__(self):
        pass
        
    def save_obj_TLSA(self, dictionary, path):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        f.close()
    
    
    def load_TLSA(self, path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    def plot_dstribution_sex(self, sex, saveloc):
        plt.figure()
        plt.ylabel('p(gender)', size=15)
        plt.title('Gender distribution (RADC-mg)', size = 15)
        print(np.sum(sex))
        
        plt.bar(['male', 'female'], [np.round(np.sum(sex)/sex.shape[0], 2), np.round(1.0 - np.sum(sex)/sex.shape[0], 2)], color = ['slateblue', 'indianred'])
        #plt.rcParams['font.size'] = 15
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    
        plt.savefig(saveloc)
        plt.close()
    def data_loader_conn(self, dataloc, path_to_save_max, saveloc, ret_subid=False):
        conn_mats = []
        age = []
        sex = []
        subids = []
        
        
        sub_details = self.load_TLSA(dataloc)
        sub_details_data = sub_details['data']
        
        
        scan_index = [0, 0, 0, 0, 0]
        for sub_id in sub_details_data.keys():
            if int(sub_id) in [13, 109, 133, 356]:
                continue
            #print(sub_details_data[sub_id])
            #break
            try:
                first_scan_index = sub_details_data[sub_id].scans.index(1)
            except ValueError:
                scan_index[-1] += 1 
                continue
            if sub_details_data[sub_id].diagnosis[first_scan_index] == 'HV':
                subids.append(int(sub_id))
                conn_mats.append([sub_details_data[sub_id].conn_mat[first_scan_index]])
                age.append(sub_details_data[sub_id].age[first_scan_index])
                sex.append(sub_details_data[sub_id].sex)
        
        age = np.array(age, dtype='float32')
        sex = np.array(sex)
        subids = np.array(subids)
        self.plot_dstribution_sex(sex, saveloc+"tlsa_distr.png")
        conn_mats = np.concatenate(conn_mats, axis=0)
        
        #print(np.sum(sex), conn_mats.shape, age)
        if ret_subid == False:
            return conn_mats, age
        else: 
            return conn_mats, age, subids