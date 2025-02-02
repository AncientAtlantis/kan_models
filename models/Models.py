import sys
import os
sys.path.append(os.path.join('..','..'))

import tensorflow as tf
from kan_models.layers.Fourier import FourierKanLayer
from kan_models.layers.MLP import MLPLayer
from kan_models.layers.Segment import SegmentKanLayer, SegmentKanLayerV2
from kan_models.layers.Spline import SplineKanLayer


type_map={'fourier':FourierKanLayer,\
          'mlp':MLPLayer,\
          'spline':SplineKanLayer,\
          'segment':SegmentKanLayer,\
          'segmentv2':SegmentKanLayerV2}


class Sequential(tf.Module):
    """
        The Sequential model composed by several layers
    """
    def __init__(self,\
                 in_size=400,\
                 n_neurons=[300,300,300,300,1],\
                 layer_types=['fourier','mlp','spline','segment','segmentv2'],\
                 layer_configs=[{\
                                    'grid_size':20,\
                                    'alpha_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'beta_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'precision':tf.float32,\
                                },\
                                {
                                    'w_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'b_initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'bias_function':tf.nn.relu,\
                                    'precision':tf.float32,\
                                },\
                                {
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
                                {
                                    'grid_size':10,\
                                    'delta':1e-8,\
                                    'initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'precision':tf.float32,\
                                    'idx_precision':tf.int32,\
                                },
                                {
                                    'grid_size':10,\
                                    'delta':1e-8,\
                                    'initializer':tf.random_normal_initializer(mean=0.0,stddev=0.1),\
                                    'precision':tf.float32,\
                                }],\
                 name_prefix='Sequential'):
        super().__init__(name=name_prefix)
        self.layers=[]
        for layer,(out_dim,config,l_type) in enumerate(zip(n_neurons,layer_configs,layer_types)):
            in_dim=in_size if layer==0 else n_neurons[layer-1]
            prefix=name_prefix+'_'+l_type+'_{:d}'.format(layer)
            self.layers.append((type_map[l_type])(in_dim,out_dim,**config,name_prefix=prefix))            
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            for li in self.layers:
                li.build()
            self.is_build=True
    
    @tf.function
    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            The forward propagation
        """
        for i,li in enumerate(self.layers):
            outs=li(inputs if i==0 else outs)

        return outs


if __name__=='__main__':
    #test block
    pass
