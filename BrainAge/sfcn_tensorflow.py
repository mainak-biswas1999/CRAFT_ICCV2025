import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
from plotter import *
from keras.models import load_model
#this code is updated to tensorflow 2 - mainak (Hang Pang code is in pytorch)
class SFCN_tf:
    def __init__(self,  channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True, pretrained=True, to_train_full=True, loc='./UKBiobank_deep_pretrain/brain_age/model_pickle/sfcn.pkl'):
        
        n_layer = len(channel_number)
        
        self.model = None
        self.to_train_enc = to_train_full
        self.model_target = None
        self.pretrained = pretrained
        self. weights_loc = loc
        self.feature_extractor = keras.Sequential(name='feature_extractor')
        self.hyperparameter = {'bs': 4,
                               'lr': 1e-4
                               }
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add(self.conv_layer(in_channel,
                                                           out_channel,
                                                           maxpool=True,
                                                           kernel_size=3,
                                                           padding='same', name='conv_%d' % i))
            else:
                self.feature_extractor.add(self.conv_layer(in_channel,
                                                            out_channel,
                                                            maxpool=False,
                                                            kernel_size=1,
                                                            padding='valid', name='conv_%d' % i))

        self.classifier = keras.Sequential(name='classifier')
        avg_shape = [5, 6, 5]
        self.classifier.add(layers.AveragePooling3D(avg_shape, name='average_pool'))
        if dropout is True:
            self.classifier.add(layers.Dropout(0.5, name='dropout'))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add(layers.Conv3D(out_channel, 1, padding='same', name='conv_%d' % i))
        self.make_model()

    def save_model(self, saveloc):
        self.model_target.save(saveloc)
        
    def load_model(self, saveloc, to_ret=False):
        self.model_target = load_model(saveloc, compile=False)
        if to_ret == True:
            return self.model
        self.model_target.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']), loss='mse')
        self.model_target.summary()

    def load_model_weighted_outs(self, saveloc):
        x = keras.Input(shape=(160, 192, 160, 1))
        temp_mod = load_model(saveloc, compile=False)
        temp_mod.compile()
        x_o = temp_mod(x)
        label = keras.Input(1, name='labels')
        weights = keras.Input(1, name='weights')
        self.model_target = keras.Model(inputs=[x, label, weights], outputs=x_o)
        self.model_target.add_loss(self.weighted_mseloss(label, x_o, weights))
        self.model_target.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']))
        self.model_target.summary()
        del temp_mod

    def weighted_mseloss(self, y_true, y_pred, weights):
        #0 -> actual labels, 1- weights
        # print(weights.shape)
        return tf.reduce_mean(weights*(y_true - y_pred)**2)    #/tf.reduce_sum(weights)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding='valid', maxpool_stride=2, name='l'):
        #only valid and same padding done in pytorch by using 1/0 -> similarly changed here
        if maxpool is True:
            layer = keras.Sequential(layers=[layers.Conv3D(out_channel, kernel_size, padding=padding),
                                            layers.BatchNormalization(),
                                            layers.MaxPool3D(pool_size=2, strides=maxpool_stride),
                                            layers.ReLU()
                                            ], name=name)

        else:
            layer = keras.Sequential(layers=[layers.Conv3D(out_channel, kernel_size, padding=padding),
                                         layers.BatchNormalization(),
                                         layers.ReLU()], name=name)
        
        return layer
    

    def softmax(self, X):
        v = np.exp(X)
        Z = np.sum(v, axis=1, keepdims=True)
        # print(v.shape, Z.shape, np.sum(v/Z, axis=1))
        return v/Z 

    def predict_orig(self, X):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.hyperparameter['bs']))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.hyperparameter['bs']


            age_pred = self.model(X[i*self.hyperparameter['bs'] : end_pos]).numpy().squeeze()
            all_ops.append(age_pred)
            
        return self.softmax(np.concatenate(all_ops))
    

    def predict_mse(self, X):
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.hyperparameter['bs']))
        
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.hyperparameter['bs']
            #print(i*self.hyperparameter['bs_pred'], end_pos)
            
            age_pred = self.model_target(X[i*self.hyperparameter['bs'] : end_pos]).numpy()
            all_ops.append(age_pred.squeeze(axis=-1))
            
        return np.concatenate(all_ops)

    def predict_mse2(self, X):
        
        all_ops = []
        n_times = int(np.ceil(X.shape[0]/self.hyperparameter['bs'])) 
        for i in range(n_times):
            if i == n_times - 1:
                end_pos = X.shape[0]
            else:
                end_pos = (i+1)*self.hyperparameter['bs']
            #print(i*self.hyperparameter['bs_pred'], end_pos)
            label = np.ones((end_pos - i*self.hyperparameter['bs'], 1))
            age_pred = self.model_target([X[i*self.hyperparameter['bs'] : end_pos], label, label]).numpy()
            all_ops.append(age_pred.squeeze(axis=-1))
            
        return np.concatenate(all_ops)

    #originally forward 
    def make_model(self):
        #out = list()
        x = keras.Input(shape=(160, 192, 160, 1))
        x_f = self.feature_extractor(x)
        x_o = self.classifier(x_f)
        self.model = keras.Model(inputs=x, outputs=x_o)
        self.model.compile()
        if self.pretrained:
            self.load_weights()
            self.summary()
        else:
            self.model.summary()
        #x = F.log_softmax(x, dim=1)
        #out.append(x)
        #return out
    
    def get_multimodal_feat_extractor(self):
        x = keras.Input(shape=(160, 192, 160, 1))
        x_f = self.feature_extractor(x, training=self.to_train_enc)
        x_f = layers.AveragePooling3D([5, 6, 5], name='average_pool')(x_f)
        x_f = layers.Reshape((x_f.shape[-1],), name='reshape')(x_f)
        return keras.Model(inputs=x, outputs=x_f)

    def add_mse_head(self, to_ret=False):
        #replace the output of the last conv layer with a mse layer
        x = keras.Input(shape=(160, 192, 160, 1))
        x_f = self.feature_extractor(x, training=self.to_train_enc)
        x_f = layers.AveragePooling3D([5, 6, 5], name='average_pool')(x_f)
        # x_f = layers.Reshape((x_f.shape[-1]*x_f.shape[-2]*x_f.shape[-3]*x_f.shape[-4],), name='reshape')(x_f)
        x_f = layers.Reshape((x_f.shape[-1],), name='reshape')(x_f)
        # x_f = layers.Dropout(0.5, name='dropout_pre')(x_f)
        # x_f = layers.Dense(64, activation='elu', name='dense1')(x_f)
        x_f = layers.Dropout(0.5, name='dropout')(x_f)
        x_o = layers.Dense(1, activation='sigmoid', name='output_layer')(x_f)
        self.model_target = keras.Model(inputs=x, outputs=x_o)
        if to_ret == True:
            return self.model_target

        self.model_target.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']), loss='mse')
        self.model_target.summary()

    def add_mse_head_weighted(self):
        #replace the output of the last conv layer with a mse layer
        x = keras.Input(shape=(160, 192, 160, 1))
        x_f = self.feature_extractor(x, training=self.to_train_enc)
        x_f = layers.AveragePooling3D([5, 6, 5], name='average_pool')(x_f)
        # x_f = layers.Reshape((x_f.shape[-1]*x_f.shape[-2]*x_f.shape[-3]*x_f.shape[-4],), name='reshape')(x_f)
        x_f = layers.Reshape((x_f.shape[-1],), name='reshape')(x_f)
        # x_f = layers.Dropout(0.5, name='dropout_pre')(x_f)
        # x_f = layers.Dense(64, activation='elu', name='dense1')(x_f)
        x_f = layers.Dropout(0.5, name='dropout')(x_f)
        x_o = layers.Dense(1, activation='sigmoid', name='output_layer')(x_f)
        label = Input(1, name='labels')
        weights = Input(1, name='weights')
        self.model_target = keras.Model(inputs=[x, label, weights], outputs=x_o)
        self.model_target.add_loss(self.weighted_mseloss(label, x_o, weights))
        self.model_target.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameter['lr']))
        self.model_target.summary()
        
    def summary(self):
        #manual dfs (sort of hardcoded as the layers are of differet types)
        beauty_len = "batch_normalization_5"
        beauty_len2 = "(None, 160, 192, 160, 32)"
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        print("Model: SFCN")
        print("________________________________________________________________________")
        print("         Layer (type)                  Output Shape            Param #")
        print("========================================================================")
        for block in self.model.layers:
            # weights, biases = layer.get_weights()
            
            if block.name[0:5]=='input':
                size_fill =  " " * (13 + len(beauty_len) - len(block.name))
                size_fill2 = " " * (4 + len(beauty_len2) - len(str(block.output_shape[0])))
                print("{}{}{}{}{}".format(block.name, size_fill, block.output_shape[0], size_fill2, 0))
                print("------------------------------------------------------------------------")
                continue

            print(block.name)
            for conv_block in block.layers:
                #if classifier - then only 3 sub layers exist
                if block.name=='feature_extractor':
                    print("    ",conv_block.name)
                    #if feature extractor each sublayer has multiple sublayers
                    for layer in conv_block.layers:
                        lname = layer.name[0:5]
                        if lname == 'batch':
                            #print([weight.name for weight in layer.weights])
                            weights = layer.get_weights()
                            n_params = int(len(weights) * np.prod(weights[0].shape))
                            trainable_params += int(n_params*0.5)
                            non_trainable_params += int(n_params*0.5)
                            # for i in range(len(weights)):
                            #     print(weights[i])
                        elif lname in ['max_p', 're_lu']:
                            n_params = 0
                        else:
                            weights, biases = layer.get_weights()
                            # print(weights.shape)
                            n_params = int(np.prod(weights.shape) + np.prod(biases.shape))
                            trainable_params += n_params
                        
                        size_fill =  " " * (4 + len(beauty_len) - len(layer.name))
                        size_fill2 = " " * (4 + len(beauty_len2) - len(str(layer.output_shape)))
                        print("         {}{}{}{}{}".format(layer.name, size_fill, layer.output_shape, size_fill2, f'{n_params:,}'))
                        total_params += n_params
                elif block.name == 'classifier':
                    #if feature extractor each sublayer has multiple sublayers
                    lname = conv_block.name[0:5]
                    if lname in ['avera', 'dropo']:
                        n_params = 0
                    else:
                        weights, biases = conv_block.get_weights()
                        # print(weights.shape)
                        n_params = int(np.prod(weights.shape) + np.prod(biases.shape))
                        trainable_params += n_params
                    size_fill =  " " * (4 + len(beauty_len) - len(conv_block.name))
                    size_fill2 = " " * (4 + len(beauty_len2) - len(str(conv_block.output_shape)))
                    print("         {}{}{}{}{}".format(conv_block.name, size_fill, conv_block.output_shape, size_fill2, f'{n_params:,}'))
                    total_params += n_params 
            
            if block.name == 'feature_extractor':
                print("------------------------------------------------------------------------")    

        print("========================================================================")
        print("Total params: ", f'{total_params:,}')
        print("Trainable params: ", f'{trainable_params:,}')
        print("Non-trainable params: ", f'{non_trainable_params:,}')
        print("________________________________________________________________________")

    def load_weights(self):
        with open(self.weights_loc, 'rb') as fptr:
            sfcn_weights_dict = pickle.load(fptr)
        # print(sfcn_weights_dict.keys())

        #manual dfs (sort of hardcoded as the layers are of differet types)
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for block in self.model.layers:
            # weights, biases = layer.get_weights()
            if block.name[0:5]=='input':
                continue
            for conv_block in block.layers:
                #if classifier - then only 3 sub layers exist
                if block.name=='feature_extractor':
                    for layer in conv_block.layers:
                        lname = layer.name[0:5]
                        if lname == 'batch':
                            #(4,) tuple of (c,) weights
                            #same order as pytorch: ['batch_normalization/gamma:0', 'batch_normalization/beta:0', 'batch_normalization/moving_mean:0', 'batch_normalization/moving_variance:0']
                            weights = sfcn_weights_dict['feature_extractor'][conv_block.name]['batch_normalization']['weight']
                            biases = sfcn_weights_dict['feature_extractor'][conv_block.name]['batch_normalization']['bias']
                            running_mean = sfcn_weights_dict['feature_extractor'][conv_block.name]['batch_normalization']['running_mean']
                            running_var = sfcn_weights_dict['feature_extractor'][conv_block.name]['batch_normalization']['running_var']
                            num_batch_processed = sfcn_weights_dict['feature_extractor'][conv_block.name]['batch_normalization']['num_batches_tracked']

                            layer.set_weights([weights, biases, running_mean, running_var])
                            n_params = int(np.prod(weights.shape) + np.prod(biases.shape)) + int(np.prod(running_mean.shape) + np.prod(running_var.shape))
                            trainable_params += int(np.prod(weights.shape) + np.prod(biases.shape))
                            non_trainable_params += int(np.prod(running_mean.shape) + np.prod(running_var.shape))
                        elif lname in ['max_p', 're_lu']:
                            n_params = 0
                        else:
                            weights = sfcn_weights_dict['feature_extractor'][conv_block.name]['conv3d']['weight']
                            biases = sfcn_weights_dict['feature_extractor'][conv_block.name]['conv3d']['bias']
                            layer.set_weights((weights, biases))
                            n_params = int(np.prod(weights.shape) + np.prod(biases.shape))
                            trainable_params += n_params
                        
                        total_params += n_params
                elif block.name == 'classifier':
                    #if feature extractor each sublayer has multiple sublayers
                    lname = conv_block.name[0:5]
                    if lname in ['avera', 'dropo']:
                        n_params = 0
                    else:
                        weights = sfcn_weights_dict['classifier']['conv_6']['weight']
                        biases = sfcn_weights_dict['classifier']['conv_6']['bias']
                        conv_block.set_weights((weights, biases))
                        n_params = int(np.prod(weights.shape) + np.prod(biases.shape))
                        trainable_params += n_params
                    total_params += n_params 
            
        print("Model loaded from pytorch weights")  
        print("Total params: ", f'{total_params:,}')
        print("Trainable params: ", f'{trainable_params:,}')
        print("Non-trainable params: ", f'{non_trainable_params:,}')

    def train_model(self, data, label, m_saveloc, r_saveloc, num_epochs=100, val_data = None, val_label = None, weights=None):
        # print(label, label.shape, data.shape)
        if not os.path.exists(m_saveloc):
            os.mkdir(m_saveloc)
        if not os.path.exists(r_saveloc):
            os.mkdir(r_saveloc)
        if weights is not None:
            #will only work if the model is compiled with add mse head weights
            history = self.model_target.fit([data, label, weights], y=None, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            plot_curves(history.history['loss'], "Training Loss (WMSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            #print(history)
            self.save_model(m_saveloc)
        #print(data.shape, label.shape, weights.shape)
        elif val_data is None:
            history = self.model_target.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'])
            #self.plot_curves(history.history['loss'], "Training Loss (BCE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            plot_curves(history.history['loss'], "Training Loss (MSE)", "Loss", num_epochs, r_saveloc+"loss.png", x_label='Epoch', x_axis_vals=[])
            #print(history)
            self.save_model(m_saveloc)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=m_saveloc, monitor='val_loss', mode='min', save_best_only=True)
            history = self.model_target.fit(data, y=label, epochs=num_epochs, batch_size=self.hyperparameter['bs'], validation_data=(val_data, val_label), callbacks=[model_checkpoint_callback])
            #accuracy plot
            #self.plot_curves2(history.history['accuracy'], history.history['val_accuracy'], "Learning Curve (Accuracy)", "Accuracy", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Acc.', 'Val Acc.'], x_axis_vals=[])
            plot_curves2(history.history['loss'], history.history['val_loss'], "Learning Curve (MSE)", "RMSE Error", num_epochs, r_saveloc+"learningcurve.png", x_label='Epoch', legends=['Train Err.', 'Val Err.'], x_axis_vals=[])

# obj = SFCN_tf()
# obj.add_mse_head()
# obj.print_all_layers()