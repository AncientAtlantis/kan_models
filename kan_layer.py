import tensorflow as tf

class SegmentKanLayer(tf.Module):
    """
    The Segment Kan layer
    """
    def __init__(self,
                 in_size=300,
                 out_size=300,
                 grid_size=10,
                 delta=1e-8,
                 initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 precision=tf.float64,
                 name_prefix='SegmentKanLayer'):
        
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.grid_size=grid_size
        self.delta=delta
        self.initializer=initializer
        self.precision=precision
        self.name_prefix=name_prefix    
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            self.coeff=tf.Variable(self.initializer([self.in_size,self.out_size,self.grid_size]),
                                   dtype=self.precision,
                                   name=self.name_prefix+'_coeff',
                                   trainable=True)
            self.is_build=True
    
    @tf.function
    def __call__(self,inputs):
        if not self.is_build:
            self.build()
        
        #The forward propagation
        #inputs with shape of (..., in_size)

        #Map inputs within (-1, 1)
        self.inputs_shape=inputs.shape
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))

        #Rescale each input to span over (-1,1)
        inputs_min=tf.math.reduce_min(inputs,axis=-1,keepdims=True)
        inputs_max=tf.math.reduce_max(inputs,axis=-1,keepdims=True)
        inputs_span=inputs_max-inputs_min
        small=tf.ones_like(inputs,dtype=self.precision)*tf.cast(self.delta,self.precision)
        denominator=tf.where(inputs_span==0.0,small,inputs_span)
        inputs=tf.cast(2.0,self.precision)*(inputs-inputs_min)/denominator-tf.cast(1.0,self.precision)
        inputs=inputs*tf.cast(1.0-self.delta,self.precision)

        #Build indices tensor: (..., in_size, out_size, 3)
        #seg_idx_l, seg_idx_h, mods: (..., in_size, out_size, 1)
        delta_l=tf.cast(2.0/(self.grid_size-1),self.precision)
        xs=inputs+tf.cast(1.0,self.precision)
        seg_idx_l=tf.math.floordiv(xs,delta_l)
        seg_idx_l=tf.cast(seg_idx_l,tf.int32)
        seg_idx_h=seg_idx_l+tf.cast(1,tf.int32)
        mods=tf.math.floormod(xs,delta_l)
        for j in range(2):
            seg_idx_h=tf.expand_dims(seg_idx_h,axis=-1)
            seg_idx_l=tf.expand_dims(seg_idx_l,axis=-1)
            mods=tf.expand_dims(mods,axis=-1)
        zeros=tf.zeros([self.in_size,1],dtype=self.precision)
        seg_idx_h=seg_idx_h+zeros
        seg_idx_l=seg_idx_l+zeros
        mods=mods+zeros
        #IN, OUT: (..., in_size, out_size, 1)
        ins=tf.range(0,self.in_size,dtype=tf.int32)
        outs=tf.range(0,self.out_size,dtype=tf.int32)
        IN,OUT=tf.meshgrid(ins,outs,indexing='ij')
        IN,OUT=tf.expand_dims(IN,axis=-1),tf.expand_dims(OUT,axis=-1)
        if len(self.inputs_shape)>1:
            for i in range(len(self.inputs_shape)-1):
                IN=tf.expand_dims(IN,axis=0)
                OUT=tf.expand_dims(OUT,axis=0)
        indices_l=tf.concatnate([IN,OUT,seg_idx_l],axis=-1)
        indices_h=tf.concatnate([IN,OUT,seg_idx_h],axis=-1)

        #Draw coeff from indices
        #matrix_l, matrix_h, matrix: (..., in_size, out_size)
        matrix_l=tf.gather_nd(self.coeff,indices_l)
        matrix_h=tf.gather_nd(self.coeff,indices_h)
        mods=tf.squeeze(mods)
        matrix=matrix_l*(tf.cast(1.0,self.precision)-mods)+\
               matrix_h*mods
        
        #Outputs: (..., out_size)
        outputs=tf.matmul(matrix,tf.expand_dims(inputs,axis=-1),transpose_a=True)
        return tf.squeeze(outputs)

class FourierKanLayer(tf.Module):
    def __init__(self,
                 out_size,
                 base_function,

                 name=None):
        super().__init__(name=None)

        pass

class SplineKanLayer(tf.Module):
    def __init__(self,
                 out_size,
                 base_function,

                 name=None):
        super().__init__(name=None)

        pass

if __name__=='__main__':
    #test block
    pass
