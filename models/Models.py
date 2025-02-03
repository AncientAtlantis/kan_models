import sys
import os
sys.path.append(os.path.join('..','..'))

import tensorflow as tf

from kan_models.layers.Fourier import FourierKanLayer
from kan_models.layers.MLP import MLPLayer
from kan_models.layers.Segment import SegmentKanLayer, SegmentKanLayerV2
from kan_models.layers.Spline import SplineKanLayer

user_layer_types=['fourier','mlp','spline','segment','segmentv2']

type_map={'fourier':FourierKanLayer,\
          'mlp':MLPLayer,\
          'spline':SplineKanLayer,\
          'segment':SegmentKanLayer,\
          'segmentv2':SegmentKanLayerV2}


default_layer_configs={'fourier':{\
                            'grid_size':20,\
                            'alpha_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'beta_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'precision':tf.float32,\
                            },\
                        'mlp':{
                            'w_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'b_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'bias_function':tf.nn.relu,\
                            'precision':tf.float32,\
                            },\
                        'spline':{
                            'grid_size':10,\
                            'k':3,\
                            'half_range':1.0,\
                            'bias_functioin':tf.nn.silu,\
                            'scale_base_trainable':True,\
                            'scale_bias_trainable':True,\
                            'scale_bias_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'scale_base_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'coeff_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'precision':tf.float32,\
                            },
                        'segment':{
                            'grid_size':10,\
                            'delta':1e-8,\
                            'initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'precision':tf.float32,\
                            'idx_precision':tf.int32,\
                            },
                        'segmentv2':{
                            'grid_size':10,\
                            'delta':1e-8,\
                            'initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                            'precision':tf.float32,\
                        }}


class Sequential(tf.Module):
    """
        The Sequential model composed by several kan layers
    """
    def __init__(self,\
                 in_size=400,\
                 n_neurons=[300,300,300,300,10],\
                 layer_types=['fourier','mlp','spline','segment','segmentv2'],\
                 layer_configs=default_layer_configs,\
                 post_ac_func=tf.nn.softmax,
                 name_prefix='Sequential'):
        super().__init__(name=name_prefix)
        self.layers=[]
        self.in_dims=[]
        self.layer_types=layer_types
        self.post_ac_func=post_ac_func
        for layer,(out_dim,l_type) in enumerate(zip(n_neurons,layer_types)):
            in_dim=in_size if layer==0 else n_neurons[layer-1]
            self.in_dims.append(in_dim)
            prefix=name_prefix+'_'+l_type+'_{:d}'.format(layer)
            cfg=layer_configs[l_type]
            self.layers.append((type_map[l_type])(in_dim,out_dim,**cfg,name_prefix=prefix))            
        self.is_build=False
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            for in_dim, lt, li in zip(self.in_dims, self.layer_types, self.layers):
                if lt in user_layer_types:
                    li.build()
                else:
                    li.build(in_dim)
            self.is_build=True

    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            The forward propagation
        """
        for i,li in enumerate(self.layers):
            outs=li(inputs if i==0 else outs)
        if self.post_ac_func:
            return self.post_ac_func(outs)
        else:
            return outs
        
    def setup(self,\
              optimizer=tf.keras.optimizer.Adam(learning_rate=1e-3),\
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CatrgoricalAccuracy()]):
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.is_compiled=True


    def __update_metrics(self,y,y_pred):
        for mi in self.metrics:
            mi.update_state(y,y_pred)
    

    def __reset_metrics(self):
        for mi in self.metrics:
            mi.reset_states()
        
    @tf.function
    def __train_step(self,x_batch,y_batch):
        #Apply gradients and update metrics
        with tf.GradientTape() as tape:
            y_pred_batch=self(x_batch)
            loss=self.loss(y_batch,y_pred_batch)
        grads=tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.__update_metrics(y_batch,y_pred_batch)
        return loss
    
    @tf.function
    def __validation_step(self,x_val_batch,y_val_batch):
        y_val_batch_pred=self(x_val_batch)
        loss=self.loss(y_val_batch,y_val_batch_pred)
        self.__update_metrics(y_val_batch,y_val_batch_pred)
        return loss
    

    def __single_epoach_loop(self,\
                             x,\
                             y,\
                             epoach=0,
                             batch_size=32,\
                             total_steps=None,\
                             apply_gradient=True,\
                             print_freq=None,\
                             messg_prefix='messg',
                             messg_container=None):
        #Single epoach over the inputs data set
        #The total steps
        if not total_steps:
            length=x.shape[0]
            round_steps,reminder=length//int(batch_size), length%int(batch_size)
            total_steps=round_steps if reminder==0 else round_steps+1
        
        #The inner loop 
        for step in range(total_steps):
            #The accumulated training steps
            acc_steps=step*int(batch_size)+total_steps*epoach

            #Draw a batch from data set
            idx_beg=step*batch_size
            x_batch,y_batch=x[idx_beg:idx_beg+batch_size],y[idx_beg:idx_beg+batch_size]
            
            #Perform a single step
            if apply_gradient:
                loss=self.__train_step(x_batch,y_batch)
            else:
                loss=self.__validation_step(x_batch,y_batch)            
            
            #Outputs key infos every print_freq steps
            if acc_steps%print_freq==0:
                messg=f'{messg_prefix}, epoach: {epoach}, current steps: {step}, accumulated steps: {acc_steps}, loss: {loss:.6f}'
                for mi in self.metrics:
                    messg_add=f', {mi.__name__}: {mi.results.numpy():.6f}'
                    messg=messg+messg_add
                print(messg)
                messg_container.append(messg)

    def train(self,\
              x=None,\
              y=None,\
              batch_size=32,\
              epoachs=10,\
              shuffle=True,\
              validation_data=None,\
              print_freq=10,\
              validation_batch_size=None,\
              validation_steps=None):
        
        #assertations of data types and shapes
        assert self.is_compiled
        assert tf.is_tensor(x) and tf.is_tensor(y)
        assert x.shape[0]==y.shape[0]
        if validation_data:
            assert len(validation_data)==2 and tf.is_tensor(validation_data[0]) and tf.is_tensor(validation_data[1])
            assert validation_data[0].shape[0]==validation_data[1].shape[0]

        #Initialization of historical training and validation messages
        self.training_messg=[]
        self.validation_messg=[]

        #Entering the global training loop
        for epoach in range(epoachs):
            #Single epoach over the data set
            if shuffle:
                x=tf.random.shuffle(x,seed=epoach)
                y=tf.random.shuffle(y,seed=epoach)
            self.__reset_metrics()
            self.__single_epoach_loop(x,y,\
                                      epoach,\
                                      batch_size,\
                                      total_steps=None,\
                                      apply_gradient=True,\
                                      print_freq=print_freq,\
                                      messg_prefix='Training step',\
                                      messg_container=self.training_messg)            
            #Single epoach over the validation data set
            if not validation_data:
                continue
            x_val,y_val=validation_data
            if shuffle:
                x_val=tf.random.shuffle(x_val,seed=epoach)
                y_val=tf.random.shuffle(y_val,seed=epoach)
            validation_batch_size=int(batch_size) if not validation_batch_size else int(validation_batch_size)
            self.__reset_metrics()
            self.__single_epoach_loop(x_val,y_val,\
                                      epoach,\
                                      validation_batch_size,\
                                      total_steps=validation_steps,\
                                      apply_gradient=False,\
                                      print_freq=print_freq,\
                                      messg_prefix='Validation step',\
                                      messg_container=self.validation_messg)

if __name__=='__main__':
    #load the data set
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    data=np.load(os.path.join('..','data_sets','mnist.npz'))
    x_test,y_test,x_train,y_train=data['x_test'],data['y_test'],data['x_train'],data['y_train']
    y_test,y_train=to_categorical(y_test),to_categorical(y_train)

    
    #define the model
    from tensorflow.keras.optimizers import Adam

