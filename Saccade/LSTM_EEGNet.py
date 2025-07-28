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
from EEGModels import *
from sklearn.mixture import GaussianMixture
matplotlib.use('Agg')



def angle_loss(y_true, y_pred):
    """
    Custom loss function for models that predict the angle on the fix-sacc-fix dataset
    Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
    Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
    Therefore we compute the absolute error of the "shorter" direction on the unit circle
    """
    #for scaled data between -1 and 1, we are using multiplication by pi
    # return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(y_pred - y_true), tf.cos(y_pred - y_true)))))
    return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(y_pred - y_true)), tf.cos(np.pi*(y_pred - y_true))))))

def polar_loss(y_true, y_pred):
    #this will return the loss when y[:, 0] -- is the prediction for radius, here we would want mse
    #and y[:, 1] -- is the prediction for theta - [-pi, pi]
    
    loss_r = tf.reduce_mean(tf.square(y_pred[:, 0] - y_true[:, 0]))
    loss_theta = tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(y_pred[:, 1] - y_true[:, 1])), tf.cos(np.pi*(y_pred[:, 1] - y_true[:, 1]))))))
    return loss_r + loss_theta


class LSTM_EEGNet:
    
    def __init__(self, type_='r', input_shape=(128, 500, 1), hidden_size=128, output_shape=1, n_tfilter=10, n_efilters=20):
        #input dimension: (N, e = 128, t = 414)
        self.hyperparameter = {
                                'lr': 1e-4,
                                'l2': 1e-3,
                                'dropout': 0.5,
                                'window_sz': 50,
                                'shift': 5,
                                'bs': 128,
                               }
        self.input_shape = input_shape
        self.n_tfilter = n_tfilter
        self.n_efilters = n_efilters
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.type_ = type_
        self.model=None
        #self.make_model()
        if type_ == 'r':
            self.loss_type = 'mse'
        elif type_ == 'r_weighted':
            self.loss_type = self.weighted_mseloss
            self.p_y_gmm = None
        elif type_ == 'theta':
            self.loss_type = angle_loss

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

    def make_separable_conv_model(self):
        inp = Input(shape=self.input_shape, name='Input_EEG')   #batch_size=self.hyperparameter['bs']) 
        #time-wise convolution
        time_conv = Conv2D(self.n_tfilter, (1, self.hyperparameter['window_sz']), strides=(1, self.hyperparameter['shift']), activation='elu', padding = 'same', name = 'CONV_time')(inp)
        time_conv = Dropout(self.hyperparameter['dropout'], name='dropout_time')(time_conv)
        
        #this will do a channel level convolution
        #separtely performs depthwise convolution
        x = None
        for i in range(self.n_efilters):
            y_temp = DepthwiseConv2D((self.input_shape[0], 1), strides=(1, 1), activation='elu', padding = 'valid', name = 'CONV_elec_f{}'.format(i))(time_conv)
            y_temp = Dropout(self.hyperparameter['dropout'], name='dropout_elec_f{}'.format(i))(y_temp)
            if x is None:
                x = y_temp
            else:
                x = tf.concat([x, y_temp], axis=1)
        
        elec_conv = Permute((2, 3, 1), name='Rearrange')(x)
        elec_conv = Reshape((elec_conv.shape[1], elec_conv.shape[2]*elec_conv.shape[3]), name='Reshape_t_features')(elec_conv)
        return Model(inputs=inp, outputs=elec_conv, name='separable_convolver')

    def weighted_mseloss(self, y_true, y_pred, weights):
        #0 -> actual labels, 1- weights
        # print(weights.shape)
        return tf.reduce_sum(weights*(y_true - y_pred)**2)    #/tf.reduce_sum(weights)
    
    def make_model(self, to_ret=False): 
        Input_eeg = Input(shape=self.input_shape, name='EEG_data') #batch_size=self.hyperparameter['bs'])
        
        #preprocess the data - using a EEGnet type of a network
        conv_model = self.make_separable_conv_model()
        #conv_model = EEGNet(Chans=self.input_shape[0], Samples=self.input_shape[1], dropoutRate=0.25, kernLength=self.hyperparameter['window_sz'], shift=self.hyperparameter['shift'], F1=self.n_tfilter, D=5, F2=self.n_efilters*10)
        prep_eeg = conv_model(Input_eeg)
        #print(prep_eeg.shape)
        
        #sequence model to get corresponding output
        lstm_out1 = LSTM(self.hidden_size, input_shape=prep_eeg.shape[1:], dropout=self.hyperparameter['dropout'], return_sequences=True, name='LSTM_1')(prep_eeg)
        lstm_out1 = BatchNormalization(name='lstm_batchnorm')(lstm_out1)
        #outputs_lstm = Dense(self.output_shape, name='op_dense', activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(self.hyperparameter['l2']))(lstm_out1)
        #outputs_lstm = LSTM(self.output_shape, input_shape=lstm_out1.shape[1:] , dropout=self.hyperparameter['dropout'], return_sequences=True, name='LSTM_2')(lstm_out1)
        #outputs_lstm = LSTM(self.output_shape, input_shape=lstm_out1.shape[1:], activation='tanh', return_sequences=True, name='LSTM_2')(lstm_out1)
        outputs_lstm = LSTM(self.output_shape, input_shape=lstm_out1.shape[1:], activation='tanh', return_sequences=False, name='LSTM_2')(lstm_out1)
        # outputs_lstm = LSTM(self.output_shape, input_shape=lstm_out1.shape[1:], activation='linear', return_sequences=False, name='LSTM_2')(lstm_out1)
        #outputs_lstm = LSTM(self.output_shape, input_shape=lstm_out1.shape[1:], activation='sigmoid', return_sequences=False, name='LSTM_2')(lstm_out1)
        
        
        
        #lstm_out1 = tf.reshape(lstm_out1, [tf.shape(lstm_out1)[0]*tf.shape(lstm_out1)[1], tf.shape(lstm_out1)[2]])
        #outputs_lstm = Dense(self.output_shape, activation='relu', name="MLP_out")(lstm_out1)
        #outputs_lstm = tf.reshape(outputs_lstm, [tf.shape(prep_eeg)[0], tf.shape(prep_eeg)[1], tf.shape(outputs_lstm)[1]])
        
        #weighted model - in case data is missing
        #Input_weights = Input(outputs_lstm.shape[1:], name='to_add_loss')
        #Input_true_labels= Input(outputs_lstm.shape[1:], name='true_labels')
        #self.model = Model(inputs=[Input_eeg, Input_true_labels, Input_weights], outputs=outputs_lstm, name='Eyeloc_predictor')
        #self.model.add_loss(self.weighted_mseloss(Input_true_labels, outputs_lstm, Input_weights))
        
        if to_ret == True:
            self.model = Model(inputs=Input_eeg, outputs=outputs_lstm, name='Eyeloc_predictor') 
            self.model.compile()
            return self.model

        #self.model.compile(optimizer=Adam(learning_rate=self.hyperparameter['lr']), metrics=['accuracy'], loss='binary_crossentropy')       #loss='mse')
        
        if self.type_ == 'r_weighted':
            label = Input(1, name='labels')
            weights = Input(1, name='weights')
            self.model = Model(inputs=[Input_eeg, label, weights], outputs=outputs_lstm, name='Eyeloc_predictor')
            self.model.add_loss(self.loss_type(label, outputs_lstm, weights))
            self.model.compile(optimizer=Adam(learning_rate=self.hyperparameter['lr']))
        else:
            self.model = Model(inputs=Input_eeg, outputs=outputs_lstm, name='Eyeloc_predictor')
            self.model.compile(optimizer=Adam(learning_rate=self.hyperparameter['lr']), loss=self.loss_type)       #loss='mse')
        
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
    
    def train_model(self, data, label, m_saveloc, r_saveloc, num_epochs=100, val_data = None, val_label = None, weights=None):
        if not os.path.exists(m_saveloc):
            os.mkdir(m_saveloc)
        if not os.path.exists(r_saveloc):
            os.mkdir(r_saveloc)
        
        # print(data.shape, label.shape)
        #currently for 1 dims y only
        if self.type_ == 'r_weighted':
            if weights is None:
                self.p_y_gmm = GaussianMixture(n_components=12, random_state=0).fit(label)
                weights = np.expand_dims(1./(1 + np.exp(self.p_y_gmm.score_samples(label))), axis=-1)
                print(np.min(weights), np.max(weights), weights.shape, label.shape)

            history = self.model.fit([data, label, weights], y=None, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            self.plot_curves(history.history['loss'], "Training Loss (WMSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            self.save_model(m_saveloc)

        #print(data.shape, label.shape, weights.shape)
        elif val_data is None:
            history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            #self.plot_curves(history.history['loss'], "Training Loss (BCE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            self.plot_curves(history.history['loss'], "Training Loss (MSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            #print(history)
            self.save_model(m_saveloc)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=m_saveloc, monitor='val_loss', mode='min', save_best_only=True)
            history = self.model.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])
            #accuracy plot
            #self.plot_curves2(history.history['accuracy'], history.history['val_accuracy'], "Learning Curve (Accuracy)", "Accuracy", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Acc.', 'Val Acc.'], x_axis_vals=[])
            self.plot_curves2(history.history['loss'], history.history['val_loss'], "Learning Curve (RMSE)", "RMSE Error", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Err.', 'Val Err.'], x_axis_vals=[])

if __name__=='__main__':
    obj = LSTM_EEGNet()
    obj.make_model()