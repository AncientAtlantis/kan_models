import tensorflow as tf


class SplineKanLayer(tf.Module):
    """
        The KAN layer with b-spline bias function
    """
    def __init__(self,
                 in_size=300,
                 out_size=300,
                 grid_size=10,
                 k=3,
                 bias_functioin=tf.nn.silu,
                 scale_base=1.0,
                 scale_bias=1.0,
                 scale_base_trainable=True,
                 scale_bias_trainable=True,
                 scale_bias_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 scale_base_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 coeff_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 precision=tf.float32,
                 name_prefix='SplineKanLayer'):
        
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.grid_size=grid_size
        self.k=k,
        self.bias_functioin=bias_functioin,
        self.scale_base=scale_base,
        self.scale_bias=scale_bias,
        self.scale_base_trainable=scale_base_trainable,
        self.scale_bias_trainable=scale_bias_trainable,
        self.scale_bias_initializer=scale_bias_initializer,
        self.scale_base_initializer=scale_base_initializer,
        self.coeff_initializer=coeff_initializer
        self.precision=precision
        self.name_prefix=name_prefix
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            self.scale_bases=tf.Variable(self.scale_base_initializer([self.in_size,self.out_size]),
                                         dtype=self.precision,
                                         name=self.name_prefix+'_scale_bases',
                                         trainable=self.scale_base_trainable)
            self.scale_biases=tf.Variable(self.scale_bias_initializer([self.in_size,self.out_size]),
                                         dtype=self.precision,
                                         name=self.name_prefix+'_scale_biases',
                                         trainable=self.scale_bias_trainable)            
            self.coeff=tf.Variable(self.coeff_initializer([self.in_size,self.out_size,self.grid_size+2*self.k]),
                                        dtype=self.precision,
                                        name=self.name_prefix+'_coeff',
                                        trainable=True)
            self.is_build=True
    
    @tf.function
    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            The forward propagation
        """
        #Map inputs within (-pi, pi)
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))*self.pi


        """
            Build the batch matrix of b-spline bias
        """
        #xs: (..., in_size, out_size, 1)
        xs=tf.expand_dims(inputs,axis=-1)+\
           tf.zeros([self.in_size,self.out_size],dtype=self.precision)
        xs=tf.expand_dims(xs,axis=-1)

        #k: (out_size, grid_size)
        k=tf.range(0,self.grid_size,dtype=self.precision)
        k=tf.tile(tf.expand_dims(k,axis=0),[self.out_size,1])
        #batch_mat_sin: (..., in_size, out_size, grid_size)        
        batch_mat_cos=tf.math.cos(xs*k)
        batch_mat_sin=tf.math.sin(xs*k)

        """
            Compose outputs 
        """
        #matrix: (..., in_size, out_size)
        matrix_alpha=tf.einsum(batch_mat_cos,self.coeff_alpha,'...ijk,ijk->...ij')
        matrix_beta=tf.einsum(batch_mat_sin,self.coeff_beta,'...ijk,ijk->...ij')

        #outputs: (..., out_size)
        outputs=tf.math.reduce_sum(matrix_alpha+matrix_beta,axis=-2)
        return outputs


if __name__=='__main__':
    #test block
    pass
