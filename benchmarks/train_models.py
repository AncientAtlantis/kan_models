import os
import sys
import copy
sys.path.append(os.path.join('..','..'))

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy    
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from kan_models.models.Models import default_layer_configs, Sequential

def train_models(path_data_set,layer_configs,layer_type,out_file,logfile):
    #load the data set
    data=np.load(path_data_set)

    x_test,y_test,x_train,y_train=data['x_test'],data['y_test'],data['x_train'],data['y_train']
    y_test,y_train=to_categorical(y_test),to_categorical(y_train)
    x_test,x_train=tf.reshape(x_test,[-1,784]),tf.reshape(x_train,[-1,784])
    x_test,x_train=tf.cast(x_test/255,tf.float32),tf.cast(x_train/255,tf.float32)
    

    #define the model
    lr_schedule=ExponentialDecay(1e-3,decay_steps=200,decay_rate=0.93,staircase=True)
    optimizer=Adam(learning_rate=lr_schedule)
    model=Sequential(in_size=784,\
                     n_neurons=[10],
                     layer_types=[layer_type],\
                     layer_configs=layer_configs,\
                     post_ac_func=tf.nn.softmax,\
                     name_prefix=f'Sequential_{layer_type}',\
                     optimizer=optimizer,\
                     loss=CategoricalCrossentropy(),\
                     metrics=[CategoricalAccuracy()])
    model.build()

    model.train(x_train,y_train,\
                batch_size=32,\
                epoachs=5,\
                shuffle=True,\
                validation_data=(x_test,y_test),\
                print_freq=10,\
                validation_batch_size=32,
                validation_steps=None)
    signatures = {
        "serving_default": model.__call__.get_concrete_function(tf.TensorSpec([None, 784], tf.float32))
    }
    tf.saved_model.save(model,out_file,signatures=signatures,options=tf.saved_model.SaveOptions(function_aliases={}))


    with open(logfile,'w') as f:
        for line in model.training_messg:
            f.write(line+'\n')
        for line in model.validation_messg:
            f.write(line+'\n')

if __name__=='__main__':
    layer_configs=copy.deepcopy(default_layer_configs)
    layer_types=['mlp','segment','segmentv2','spline','fourier']

    data_set_path=os.path.join('..','data_sets','mnist.npz')

    for i in range(10):
        for layer_type in layer_types:
            train_models(data_set_path,layer_configs,layer_type,f'{layer_type}_{i:d}_saved_model',f'{layer_type}_{i:d}_log')

