import numpy as np
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dropout, Permute, DepthwiseConv2D, Reshape, Flatten, LSTM, BatchNormalization, Dense, Activation, MaxPool2D, ReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib

from sklearn.mixture import GaussianMixture
from read_data import *
from plotter import *
import tfimm

matplotlib.use('Agg')

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

def get_rmse1(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y.ravel(), y_pred.ravel()))

class Attention_linearCombination(tf.keras.layers.Layer):
    def __init__(self, out_size, intermediate_attn_dim, activation=None, input_sizes = [], return_weights=False, *args, **kwargs):
        #input size should have the number of features in each modality - works for 1d feature vectors only.
        super(Attention_linearCombination, self).__init__(*args, **kwargs)
        self.out_size = out_size
        self.int_dim = intermediate_attn_dim
        self.input_sizes = input_sizes
        self.activation = activation
        self.return_weights = return_weights
        
    def build(self, input_shape):
        #make the attention weights
        self.attention_weights = []
        self.attention_biases = []
        
        self.transform_weights = []
        self.transform_biases = []
        for i in range(len(self.input_sizes)):
            self.attention_weights.append(self.add_weight('kernel_att{}'.format(i+1), shape=[self.input_sizes[i], self.int_dim], initializer='glorot_uniform', trainable=True))
            self.attention_biases.append(self.add_weight('bias_att{}'.format(i+1), shape=[1, self.int_dim], initializer='glorot_uniform', trainable=True))

            if self.return_weights == False:
                self.transform_weights.append(self.add_weight('kernel_trans{}'.format(i+1), shape=[self.input_sizes[i], self.out_size], initializer='glorot_uniform', trainable=True))
                self.transform_biases.append(self.add_weight('bias_trans{}'.format(i+1), shape=[1, self.out_size], initializer='glorot_uniform', trainable=True))
        
        self.attn_vector = self.add_weight('bias', shape=[self.int_dim, 1], initializer='glorot_uniform', trainable=True)
        
    def call(self, inputs):        
        #computing the attention parameters
        #input is the list of all the modalities
        att_weights = []
        for i in range(len(self.input_sizes)):
            att_weights.append(tf.keras.activations.tanh(self.attention_biases[i] + inputs[i] @ self.attention_weights[i]) @ self.attn_vector)
            # print(tf.keras.activations.tanh(self.attention_biases[i] + inputs[i] @ self.attention_weights[i]).shape, inputs[i].shape)
            # print(att_weights[-1].shape)

        att_weights = tf.math.exp(tf.concat(att_weights, axis=-1))
        soft_att_weights = att_weights/tf.math.reduce_sum(att_weights, axis=-1, keepdims=True)
        # print(soft_att_weights.shape)
        output = 0
        if self.return_weights == True:
            return soft_att_weights

        for i in range(len(self.input_sizes)):
            transform_i = self.transform_biases[i] + inputs[i] @ self.transform_weights[i]
            output += tf.expand_dims(soft_att_weights[:, i], axis=-1) * transform_i
            # print(tf.expand_dims(soft_att_weights[:, i], axis=-1).shape, transform_i.shape)
        return Activation(self.activation)(output)


class Base_CNN:
    def __init__(self):
        pass

    def conv_layer(self, kernel_size, out_channel, name, maxpool=True, padding='same', maxpool_dims=(3,3)):
        #only valid and same padding done in pytorch by using 1/0 -> similarly changed here
        layer = tf.keras.Sequential(layers=[Conv2D(out_channel, kernel_size, padding=padding),
                                        BatchNormalization(),
                                        MaxPool2D(pool_size=maxpool_dims),
                                        ReLU()
                                        ], name=name)
        
        return layer
    
    def get_model(self):
        feature_extractor = tf.keras.Sequential(name='feature_extractor')
        feature_extractor.add(self.conv_layer((9, 9), 16, name='l1'))
        feature_extractor.add(self.conv_layer((7, 7), 32, name='l2'))
        feature_extractor.add(self.conv_layer((5, 5), 48, name='l3'))
        feature_extractor.add(self.conv_layer((3, 3), 64, name='l4'))
        feature_extractor.add(Flatten(name='Flatten'))
        
        return feature_extractor


class IntToFloat32(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        #will divide the data by 255.0
        return tf.cast(inputs, tf.float32)/255.0

class resnet_model:
    
    def __init__(self, type_='r', input_shape=(250, 250, 3), output_shape=1):
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
        self.type_ = type_
        self.loss_type = 'mse'   #'mse'   #'mae'
 

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

    def weighted_mseloss(self, y_true, y_pred, weights):
        #0 -> actual labels, 1- weights
        # print(weights.shape)
        return tf.reduce_mean(weights*(y_true - y_pred)**2) 

    def load_model_weighted_outs(self, saveloc):
        x = Input(shape=(256, 256, 3))
        temp_mod = load_model(saveloc, compile=False)
        temp_mod.compile()
        x_o = temp_mod(x)
        label = Input(1, name='labels')
        weights = Input(1, name='weights')
        self.model = Model(inputs=[x, label, weights], outputs=x_o)
        self.model.add_loss(self.weighted_mseloss(label, x_o, weights))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']))
        self.model.summary()
        del temp_mod


    def make_model(self, N_imgs=16, to_ret=False): 
        # inp = Input(shape=(N_imgs, 500, 500, 3))
        inp = Input(shape=(256, 256, 3), dtype=tf.uint8)
        out = IntToFloat32()(inp)
        # resnet = tf.keras.applications.ResNet50(weights='imagenet', pooling='avg', input_shape=(500, 500, 3), include_top=False)
        resnet = tf.keras.applications.ResNet152V2(weights='imagenet', pooling='avg', input_shape=(256, 256, 3), include_top=False)

        # resnet = Base_CNN().get_model()
        # print(resnet)
        # resnet.summary()
        # out = []
        # for i in range(N_imgs):
        #     # out.append([resnet(inp[:, i, :, :, :])])
        #     out.append(resnet(inp[:, i, :, :, :]))



        # attn_layer = Attention_linearCombination(64, 16, input_sizes = np.ones(16, dtype='int')*2048, return_weights = False,
        #                                          activation='relu', name='attention_linarComb')

        # out = attn_layer(out)
        # out = tf.concat(out, axis=0)
        # print(out.shape)
        # out = tf.math.reduce_mean(out, axis=0)
        # print(out.shape)
        
        out = resnet(out)
        out = Dropout(self.hyperparameter['dropout'], name='drp_out1')(out)
        out = Dense(64, name='Dense_l1', activation='relu')(out)
        out = Dropout(self.hyperparameter['dropout'], name='drp_out2')(out)
        out = Dense(self.output_shape, name='output', activation='sigmoid')(out)

        self.model = Model(inputs=inp, outputs=out, name='resnet-peopleCount')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']), loss=self.loss_type)
        if to_ret == True:
            return self.model
        self.model.summary()


    def predict(self, X):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.hyperparameter['bs']))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.hyperparameter['bs']
             
            if self.type_ == 'r_weighted':
                label = np.ones((end_pos - i*self.hyperparameter['bs'], 1))
                op_i = self.model([X[i*self.hyperparameter['bs'] : end_pos], label, label]).numpy()
                # print(op_i.shape)
                all_ops.append(op_i)
            else:
                all_ops.append(self.model(X[i*self.hyperparameter['bs'] : end_pos]).numpy())
        
        return np.concatenate(all_ops)
    
    def train_model(self, data, label, m_saveloc, r_saveloc, num_epochs=25, val_data = None, val_label = None, weights=None):
        if not os.path.exists(m_saveloc):
            os.mkdir(m_saveloc)
        if not os.path.exists(r_saveloc):
            os.mkdir(r_saveloc)
        
        # currently for 1 dims y only
        # print(data.shape, label.shape, weights.shape)
       	

        if self.type_ == 'r_weighted':
            val_weights = np.ones(val_label.shape).astype('float32')
            
            def generator():
                for inp, lab, wts in zip(data, label, weights):
                    yield {'input_1': inp, 
                           'labels': lab,
                           'weights': wts,
                          }, {'resnet-peopleCount': None,
                             }

            train_dataset = tf.data.Dataset.from_generator(generator, output_types=({'input_1': tf.uint8, 'labels': tf.float32, 'weights': tf.float32}, {'resnet-peopleCount': tf.float32}))

            # Shuffle and batch the data
            train_dataset = train_dataset.shuffle(buffer_size=10000)
            train_dataset = train_dataset.batch(batch_size=self.hyperparameter['bs'])


            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=m_saveloc, monitor='val_loss', mode='min', save_best_only=True)
            # history = self.model.fit([data, label, weights], y=None, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=([val_data, val_label, val_weights], None), callbacks=[model_checkpoint_callback])
            history = self.model.fit(train_dataset, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=([val_data, val_label, val_weights], None), callbacks=[model_checkpoint_callback])
            self.plot_curves(history.history['loss'], "Training Loss (WMSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            self.save_model(m_saveloc)

        # currently may fail if val data is not passed as loaders not implemented
        elif val_data is None:
            history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            self.plot_curves(history.history['loss'], "Training Loss (MSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            self.save_model(m_saveloc)
        else:
            def generator():
                for inp, lab in zip(data, label):
                    yield {'input_1': inp, 
                          }, {'output': lab,
                             }

            train_dataset = tf.data.Dataset.from_generator(generator, output_types=({'input_1': tf.uint8}, {'output': tf.float32}))

            # Shuffle and batch the data
            train_dataset = train_dataset.shuffle(buffer_size=10000)
            train_dataset = train_dataset.batch(batch_size=self.hyperparameter['bs'])

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=m_saveloc, monitor='val_loss', mode='min', save_best_only=True)
            # history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])
            history = self.model.fit(train_dataset, epochs=num_epochs, validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])           
            self.plot_curves2(history.history['loss'], history.history['val_loss'], "Learning Curve (MSE)", "MSE Error", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Err.', 'Val Err.'], x_axis_vals=[])



def call_k_fold_paper(check_train=True):
    n_epochs = 15
    n_fold = 5
    base_rloc = './results/cam17_base_patched_tl/'
    base_mloc = './models/cam17_base_patched_tl/'
    source_loc = './models/cam16_base_patched/'
    
    
    # X_train, y_train, X_val, y_val, X_test, y_test = read_cam('./data/cam16.npz', True)
    X_train, y_train, X_val, y_val, X_test, y_test = read_cam('/data1/mainak/cancer_pred/data/cam17.npz')
    
    obj_model = resnet_model()
    # obj_model.make_model()
    obj_model.load_model(source_loc)
    # obj_model.add_mse_head()
    obj_model.train_model(X_train, y_train, base_mloc, base_rloc, num_epochs=n_epochs, val_data = X_val, val_label = y_val) 
    
    #plot training curves
    pred_y_train = obj_model.predict(X_train)
    pred_y_val = obj_model.predict(X_val)
    pred_y_test = obj_model.predict(X_test)
    
    
    del obj_model
    plot_predictions_paper(pred_y_train, y_train, base_rloc+"train.png", title_="Train", addon="Coverage")
    plot_predictions_paper(pred_y_val, y_val, base_rloc+"val.png", title_="Validation", addon="Coverage")
    plot_predictions_paper(pred_y_test, y_test, base_rloc+"overall_results_paper2.png", title_="Resnet152v2: Cam17 (TL)", addon="Coverage")
    

if __name__=='__main__':
    import sys
    call_k_fold_paper(check_train=False)
    
    
    

    