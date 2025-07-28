import numpy as np
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dropout, Permute, DepthwiseConv2D, Reshape, Flatten, LSTM, BatchNormalization, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib

from sklearn.mixture import GaussianMixture
from read_data import *
from plotter import *
import tfimm


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=34000)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def predict_global(y, preds, ids):
    ids = np.squeeze(ids)
    pred = []
    act = []
    for uid in np.unique(ids):
        locs = np.where(ids == uid)[0]
        pred.append(np.sum(preds[locs]))
        act.append(np.sum(y[locs]))

        # print(pred[-1], act[-1])

    act = np.array(act)
    pred = np.array(pred)

    return np.expand_dims(act, axis=-1), np.expand_dims(pred, axis=-1)

class vit_model:
    
    def __init__(self, input_shape=(384, 384, 3), output_shape=1):
        #input dimension: (N, e = 128, t = 414)
        self.hyperparameter = {
                                'lr': 1e-4,
                                'l2': 1e-3,
                                'dropout': 0.2,
                                'window_sz': 50,
                                'shift': 5,
                                'bs': 32,
                               }

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.loss_type = 'mse'
 

    def plot_curves(self, loss, __title__, y_label, n_epochs, saveloc, x_label='Epoch', x_axis_vals=[]):
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
    
    def plot_curves2(self, loss1, loss2, __title__, y_label, n_epochs, saveloc, x_label='Epoch', legends=[], x_axis_vals=[]):
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
    
    
    def save_model(self, saveloc):
        self.model.save(saveloc)
        
    def load_model(self, saveloc, to_ret=False):
        self.model = load_model(saveloc, compile=False)
        if to_ret == True:
            return self.model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']), loss=self.loss_type)
        self.model.summary()

    def make_model(self, to_ret=False): 
        # model = tfimm.create_model("vit_tiny_patch16_224", pretrained="timm")
        inp = Input(shape=(384, 384, 3))
        vit_base_p32_384 = tfimm.create_model("vit_base_patch16_384", pretrained="timm")  #training=True)
        # ViT-B_32
        # model.summary()
        # print(tfimm.list_models(pretrained="timm"))
        # 
        # resnet = tf.keras.applications.VGG16(weights='imagenet', pooling='avg', input_shape=(384, 384, 3), include_top=False)
        # # resnet = tf.keras.applications.ResNet152V2(weights='imagenet', pooling='avg', input_shape=(384, 384, 3), include_top=False)
        out = vit_base_p32_384(inp)
        out = Dropout(self.hyperparameter['dropout'], name='drp_out1')(out)
        # out = Dense(64, name='Dense_l1', activation='relu')(out)
        # out = Dropout(self.hyperparameter['dropout'], name='drp_out2')(out)
        out = Dense(self.output_shape, name='output', activation='sigmoid')(out)

        self.model = Model(inputs=inp, outputs=out, name='resnet-peopleCount')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']), loss=self.loss_type)
        self.model.summary()

    def predict(self, X):
        # preprocess = tfimm.create_preprocessing("vit_base_patch32_384", dtype="float32")
        # X = preprocess((X*255).astype('int'))

        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.hyperparameter['bs']))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.hyperparameter['bs']
            
            all_ops.append(self.model(X[i*self.hyperparameter['bs'] : end_pos]).numpy())
        
        return np.concatenate(all_ops)
    
    def train_model(self, data, label, m_saveloc, r_saveloc, num_epochs=50, val_data = None, val_label = None):
        # preprocess = tfimm.create_preprocessing("vit_base_patch32_384", dtype="float32")
        # data = preprocess((data*255).astype('int'))
        
        if not os.path.exists(m_saveloc):
            os.mkdir(m_saveloc)
        if not os.path.exists(r_saveloc):
            os.mkdir(r_saveloc)
        
        # currently for 1 dims y only
        # print(data.shape, label.shape, weights.shape)
        def generator():
            for inp, lab in zip(data, label):
                yield {'input_1': inp, 
                      }, {'output': lab,
                         }

        train_dataset = tf.data.Dataset.from_generator(generator, output_types=({'input_1': tf.float32}, {'output': tf.float32}))

        # Shuffle and batch the data
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(batch_size=self.hyperparameter['bs'])

        if val_data is None:
            history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            self.plot_curves(history.history['loss'], "Training Loss (MSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            self.save_model(m_saveloc)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=m_saveloc, monitor='val_loss', mode='min', save_best_only=True)
            # history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])
            history = self.model.fit(train_dataset, epochs=num_epochs, validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])           
            self.plot_curves2(history.history['loss'], history.history['val_loss'], "Learning Curve (MSE)", "MSE Error", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Err.', 'Val Err.'], x_axis_vals=[])





if __name__=="__main__":
    vit_obj = vit_model()
    vit_obj.make_model()
    r_saveloc = './results/vit_subImg/'
    m_saveloc = './models/vit_subImg/'
    # X_train_id, y_train_id, X_val_id, y_val_id, X_test_id, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, rescale_obj = load_id_ood_people()
    X_train_id, y_train_id, X_val_id, y_val_id, ids_val_id, X_test_id, X_train_ood, y_train_ood, X_val_ood, y_val_ood, X_test_ood, y_test_ood, ids_test_ood, rescale_obj = load_id_ood_people(log_scale=True)
    vit_obj.train_model(X_train_id*255.0, y_train_id, m_saveloc, r_saveloc, num_epochs=10, val_data=X_val_id*255.0, val_label=y_val_id)
    # vit_obj.load_model(m_saveloc)
    y_train = np.exp(rescale_obj.inv_scale(y_train_id)) - 1.0
    y_val = np.exp(rescale_obj.inv_scale(y_val_id)) - 1.0
    # y_train = rescale_obj.inv_scale(y_train_ood)
    # y_val = rescale_obj.inv_scale(y_val_ood)
    # y_test = rescale_obj.inv_scale(y_test_ood)

    pred_train_y = np.exp(rescale_obj.inv_scale(vit_obj.predict(X_train_id))) - 1.0
    pred_val_y = np.exp(rescale_obj.inv_scale(vit_obj.predict(X_val_id))) - 1.0

    y_val, pred_val_y = predict_global(y_val, pred_val_y, ids_val_id)

    print(pred_train_y.shape, y_train.shape)
    
    

    print(y_val.shape, pred_val_y.shape)
    # pred_train_y = rescale_obj.inv_scale(obj.predict(X_train_ood))
    # pred_val_y = rescale_obj.inv_scale(obj.predict(X_val_ood))
    # pred_test_y = rescale_obj.inv_scale(obj.predict(X_test_ood))

    plot_predictions_paper(pred_train_y[:, 0], y_train[:, 0], r_saveloc+"train_r_target_paper.png", title_='Train') #, addon=" (log scale)")
    plot_predictions_paper(pred_val_y[:, 0], y_val[:, 0], r_saveloc+"val_r_target_paper_full.png", title_= 'Validation') #, addon=" (log scale)")

