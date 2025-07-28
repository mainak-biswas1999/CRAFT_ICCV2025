import tensorflow as tf
# from utils.utils import *
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from abc import ABC, abstractmethod
import tensorflow.keras as keras
import logging
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib
import tensorflow.keras as keras
from keras.callbacks import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import os

def angle_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(np.pi*(y_pred - y_true)), tf.cos(np.pi*(y_pred - y_true))))))


class BaseNet:
    def __init__(self, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0, learning_rate=1e-4):
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss 
        self.nb_channels = input_shape[1]
        self.timesamples = input_shape[0]
        self.model = self._build_model()

        # Compile the model depending on the task 
        if self.loss == 'bce':
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
        elif self.loss == 'mse':
            self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['mean_squared_error'])
        elif self.loss == 'angle-loss':
            self.model.compile(loss=angle_loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        else:
            raise ValueError("Choose valid loss for your task")
            
        # if self.verbose:
            # self.model.summary()

        logging.info(f"Number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])}")
        logging.info(f"Number of non-trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])}")

    # abstract method
    def _split_model(self):
        pass

    # abstract method
    def _build_model(self):
        pass

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save(path)

    def fit(self, X_train, y_train, X_val, y_val):
        #csv_logger = CSVLogger(config['batches_log'], append=True, separator=';')
        #ckpt_dir = config['model_dir'] + '/best_models/' + config['model'] + '_nb_{}_'.format(self.model_number) + 'best_model.h5'
        #ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_dir, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        hist = self.model.fit(X_train, y_train, verbose=2, batch_size=self.batch_size, validation_data=(X_val, y_val),
                                  epochs=self.epochs, callbacks=[early_stop])

    def predict(self, testX):
        return self.model.predict(testX)



class ConvNet(ABC, BaseNet):
    def __init__(self, input_shape, output_shape, loss, kernel_size=32, nb_filters=32, verbose=True, batch_size=64, 
                use_residual=False, depth=6, epochs=2, preprocessing = False, model_number=0):
        self.use_residual = use_residual
        self.depth = depth
        self.callbacks = None
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.preprocessing = preprocessing
        self.input_shape = input_shape
        super(ConvNet, self).__init__(input_shape=input_shape, output_shape=output_shape, loss=loss, epochs=epochs, verbose=verbose, model_number=model_number)
        logging.info('Parameters of {}, model number {}: '.format(self, model_number))
        logging.info('--------------- use residual : ' + str(self.use_residual))
        logging.info('--------------- depth        : ' + str(self.depth))
        logging.info('--------------- batch size   : ' + str(self.batch_size))
        logging.info('--------------- kernel size  : ' + str(self.kernel_size))
        logging.info('--------------- nb filters   : ' + str(self.nb_filters))
        logging.info('--------------- preprocessing: ' + str(self.preprocessing))

    # abstract method
    def _preprocessing(self, input_tensor):
        pass

    # abstract method
    def _module(self, input_tensor, current_depth):
        pass

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def _build_model(self, X=[]):
        input_layer = tf.keras.layers.Input(self.input_shape)

        if self.preprocessing:
            preprocessed = self._preprocessing(input_layer)
            x = preprocessed
            input_res = preprocessed
        else:
            x = input_layer
            input_res = input_layer

        for d in range(self.depth):
            x = self._module(x, d)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        if self.loss == 'bce':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='sigmoid')(gap_layer)
        elif self.loss == 'mse':
            # output_layer = tf.keras.layers.Dense(self.output_shape, activation='linear')(gap_layer)
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='tanh')(gap_layer)
        elif self.loss == 'angle-loss':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='tanh')(gap_layer) 
        else:
            raise ValueError("Choose valid loss function")

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model


class PyramidalCNN(ConvNet):
    """
    The Classifier_PyramidalCNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """
    def __init__(self, loss, batch_size, input_shape, output_shape, model_number=0, kernel_size=16, epochs=50, nb_filters=16, verbose=True,
                    use_residual=False, depth=6):

        super(PyramidalCNN, self).__init__(input_shape=input_shape, output_shape=output_shape, loss=loss, kernel_size=kernel_size, epochs=epochs, nb_filters=nb_filters,
                    verbose=verbose, batch_size=batch_size, use_residual=use_residual, depth=depth, model_number=model_number)

    def __str__(self):
        return self.__class__.__name__
        
    def _module(self, input_tensor, current_depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        """
        x = tf.keras.layers.Conv1D(filters=self.nb_filters*(current_depth + 1), kernel_size=self.kernel_size, padding='same', use_bias=False)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
        return x


class SpyrCNN_wrapper:
    def __init__(self, type_='r', input_shape=(128, 500), hidden_size=128, output_shape=1, n_tfilter=10, n_efilters=20):
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
    
    def make_model(self, to_ret=False): 
        pyrcnn_obj = PyramidalCNN(self.loss_type, self.hyperparameter['bs'], self.input_shape, self.output_shape)
        self.model = pyrcnn_obj.model

        if to_ret == True: 
            self.model.compile()
            return self.model

        self.model.compile(optimizer=Adam(learning_rate=self.hyperparameter['lr']), loss=self.loss_type)
        self.model.summary()
        
    def predict(self, X):
        X = np.squeeze(X)
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
        data = np.squeeze(data)
        print(data.shape)
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

if __name__ == '__main__':
    m_obj = SpyrCNN_wrapper(type_='r')
    m_obj.make_model()