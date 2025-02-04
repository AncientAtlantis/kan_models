import tensorflow as tf


class FourierKanLayer(tf.Module):
    """
        The KAN layer with fourier bias function
    """
    def __init__(self,
                 in_size=300,
                 out_size=300,
                 grid_size=10,
                 alpha_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 beta_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 precision=tf.float32,
                 name_prefix='FourierKanLayer'):
        
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.grid_size=grid_size
        self.alpha_initializer=alpha_initializer
        self.beta_initializer=beta_initializer
        self.precision=precision
        self.name_prefix=name_prefix    
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            #coeff_alpha: (in_size, out_size, grid_size)
            #coeff_beta: (in_size, out_size, grid_size)
            self.coeff_alpha=tf.Variable(self.alpha_initializer([self.in_size,self.out_size,self.grid_size]),
                                         dtype=self.precision,
                                         name=self.name_prefix+'_coeff_alpha',
                                         trainable=True)
            self.coeff_beta=tf.Variable(self.beta_initializer([self.in_size,self.out_size,self.grid_size]),
                                        dtype=self.precision,
                                        name=self.name_prefix+'_coeff_beta',
                                        trainable=True)
            self.pi=tf.constant(3.1415926,dtype=self.precision)
            self.is_build=True

    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            The forward propagation
        """

        #t0=tf.timestamp()
        #Map inputs within (-pi, pi)
        inputs=tf.cast(inputs,self.precision)
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))*self.pi
        
        #t1=tf.timestamp()  
        """
            Build the batch matrix of fourier bias
        """
        #xs: (..., in_size, out_size, 1)
        xs=tf.expand_dims(inputs,axis=-1)+\
           tf.zeros([self.in_size,self.out_size],dtype=self.precision)
        xs=tf.expand_dims(xs,axis=-1)

        #k: (out_size, grid_size)
        k=tf.range(0,self.grid_size,dtype=self.precision)
        k=tf.tile(tf.expand_dims(k,axis=0),[self.out_size,1])
        #batch_mat_sin, batch_mat_cos: (..., in_size, out_size, grid_size)        
        batch_mat_cos=tf.math.cos(xs*k)
        batch_mat_sin=tf.math.sin(xs*k)

        #t2=tf.timestamp()  
        """
            Compose outputs 
        """
        #matrix: (..., in_size, out_size)
        #matrix_alpha=tf.einsum('...ijk,ijk->...ij',batch_mat_cos,self.coeff_alpha)
        #matrix_beta=tf.einsum('...ijk,ijk->...ij',batch_mat_sin,self.coeff_beta)
        matrix_alpha=tf.reduce_sum(batch_mat_cos*self.coeff_alpha,axis=-1)
        matrix_beta=tf.reduce_sum(batch_mat_sin*self.coeff_beta,axis=-1)
        #t3=tf.timestamp()
        #outputs: (..., out_size)
        outputs=tf.math.reduce_sum(matrix_alpha+matrix_beta,axis=-2)
        #t4=tf.timestamp()

        #tf.print('total')
        #tf.print(t4-t0)
        #tf.print('inputs')
        #tf.print(t1-t0)
        #tf.print('build batch')
        #tf.print(t2-t1)
        #tf.print('computes')
        #tf.print(t3-t2)
        #tf.print('outputs')
        #tf.print(t4-t3)
        #tf.print()

        return outputs


if __name__=='__main__':
    #test block
    pass
