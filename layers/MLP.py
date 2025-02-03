import tensorflow as tf


class MLPLayer(tf.Module):
    """
        The convensional MLP layer 
    """
    def __init__(self,
                 in_size=300,
                 out_size=300,
                 w_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 b_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 bias_function=tf.nn.relu,
                 precision=tf.float32,
                 name_prefix='MLPLayer'):
        
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.w_initializer=w_initializer
        self.b_initializer=b_initializer
        self.bias_function=bias_function
        self.precision=precision
        self.name_prefix=name_prefix    
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            #coeff_beta: (in_size, out_size, grid_size)
            self.w=tf.Variable(self.w_initializer([self.in_size,self.out_size]),
                               dtype=self.precision,
                               name=self.name_prefix+'weight',
                               trainable=True)
            self.b=tf.Variable(self.b_initializer([self.out_size]),
                               dtype=self.precision,
                               name=self.name_prefix+'bias',
                               trainable=True)
            self.is_build=True

    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            The forward propagation
        """
        #inputs: (..., in)
        #w: (in, out)
        inputs=tf.cast(inputs,self.precision)
        hidden=tf.nn.bias_add(tf.matmul(inputs,self.w),self.b)
        return self.bias_function(hidden)


if __name__=='__main__':
    #test block
    pass
