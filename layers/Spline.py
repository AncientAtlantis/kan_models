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
                 half_range=1.0,
                 bias_functioin=tf.nn.silu,
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
        self.k=k
        self.half_range=half_range
        self.bias_functioin=bias_functioin
        self.scale_base_trainable=scale_base_trainable
        self.scale_bias_trainable=scale_bias_trainable
        self.scale_bias_initializer=scale_bias_initializer
        self.scale_base_initializer=scale_base_initializer
        self.coeff_initializer=coeff_initializer
        self.precision=precision
        self.name_prefix=name_prefix
        self.is_build=False
    
    def build(self):
        #self.scale_biases, self.scale_bases, self.coeff, self.grids: (in, out), (in, out), (in, out, grid_size-1+k), (in, out, grid_size+2k)
        if not self.is_build:
            self.scale_bases=tf.Variable(self.scale_base_initializer([self.in_size,self.out_size]),
                                         dtype=self.precision,
                                         name=self.name_prefix+'_scale_bases',
                                         trainable=self.scale_base_trainable)
            self.scale_biases=tf.Variable(self.scale_bias_initializer([self.in_size,self.out_size]),
                                         dtype=self.precision,
                                         name=self.name_prefix+'_scale_biases',
                                         trainable=self.scale_bias_trainable)            
            self.coeff=tf.Variable(self.coeff_initializer([self.in_size,self.out_size,self.grid_size-1+self.k]),
                                        dtype=self.precision,
                                        name=self.name_prefix+'_coeff',
                                        trainable=True)
            l_interval=2*self.half_range/(self.grid_size-1)
            grids=tf.linspace(-self.half_range-self.k*l_interval,self.half_range+self.k*l_interval,self.grid_size+self.k*2)
            grids=tf.cast(grids,self.precision)
            grids=tf.expand_dims(tf.expand_dims(grids,axis=0),axis=0)
            self.grids=tf.tile(grids,[self.in_size,self.out_size,1])
            self.is_build=True
    
    def build_B_batch(self,xs,grids,k,epsilon=1e-8):
        xs=tf.expand_dims(xs,axis=-1)
        #grids: (0, in_dims, G+2k+1)
        grids=tf.expand_dims(grids,axis=0)
        if k==0:
            values=tf.cast((xs>=grids[:,:,:-1]) & (xs<grids[:,:,1:]),self.precision)
        else:
            B_kml=self.build_B_batch(xs[:,:,0],grids[0],k-1)
            values=B_kml[:,:,:-1]*(xs-grids[:,:,:-(k + 1)])/(grids[:,:,k:-1]-grids[:,:,:-(k+1)]+epsilon)+B_kml[:,:,1:]*(grids[:,:,k+1:]-xs)/(grids[:,:,k+1:]-grids[:,:,1:-k]+epsilon)
        #values: (in, out, grid_size-1+k)
        return values

    @tf.function
    def __call__(self,inputs):
        if not self.is_build:
            self.build()


        """
            Scale the inputs
        """
        #Map inputs within (-half_range, half_range)
        #inputs: (..., in)
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))*self.half_range        


        """
            Build the batch matrix of b-spline bias
        """
        #mat: (in, out, grid_size-1+k)
        mat=self.build_B_batch(inputs,self.grids,self.k)


        """
            Compose outputs 

        """
        #values, v_base: (in, out)
        values=tf.einsum(mat,self.coeff,'ijk,ijk->ij')
        v_base=values*self.scale_bases
        #v_biases: (..., in, out)
        inputs=tf.expand_dims(self.bias_functioin(inputs),axis=-1) #(..., in, 1)
        v_biases=inputs*self.scale_biases
        #return: (..., out)
        return tf.math.reduce_sum(v_base+v_biases,axis=-2)
        

if __name__=='__main__':
    #test block
    pass
