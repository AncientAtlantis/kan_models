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
                 precision=tf.float32,
                 idx_precision=tf.int32,
                 order=0,
                 name_prefix='SegmentKanLayer'):
        
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.grid_size=grid_size
        self.delta=delta
        self.initializer=initializer
        self.precision=precision
        self.idx_precision=idx_precision
        self.order=order
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


        """
            The forward propagation
        """
        #Map inputs within (-1, 1)
        #inputs: (..., in_size)
        inputs=tf.cast(inputs,self.precision)
        self.inputs_shape=inputs.shape
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))


        """
            Rescale each input to span over (-1,1)
        """
        inputs_min=tf.math.reduce_min(inputs,axis=-1,keepdims=True)
        inputs_max=tf.math.reduce_max(inputs,axis=-1,keepdims=True)
        inputs_span=inputs_max-inputs_min
        small=tf.ones_like(inputs,dtype=self.precision)*tf.cast(self.delta,self.precision)
        denominator=tf.where(inputs_span==0.0,small,inputs_span)
        inputs=tf.cast(2.0,self.precision)*(inputs-inputs_min)/denominator-tf.cast(1.0,self.precision)
        inputs=inputs*tf.cast(1.0-self.delta,self.precision)


        """
            Build indices tensor: (..., in_size, out_size, 3)
            The indices were used by tf.gather_nd() to index from the coeff tensor 
        """
        #seg_idx_l, seg_idx_h, mods: (..., in_size, out_size, 1)
        delta_l=tf.cast(2.0/(self.grid_size-1),self.precision)
        xs=inputs+tf.cast(1.0,self.precision)
        seg_idx_l=tf.math.floordiv(xs,delta_l)
        seg_idx_l=tf.cast(seg_idx_l,self.idx_precision)
        seg_idx_h=seg_idx_l+tf.cast(1,self.idx_precision)
        mods=tf.math.floormod(xs,delta_l)
        for j in range(2):
            seg_idx_h=tf.expand_dims(seg_idx_h,axis=-1)
            seg_idx_l=tf.expand_dims(seg_idx_l,axis=-1)
            mods=tf.expand_dims(mods,axis=-1)
        zeros=tf.zeros([self.in_size,self.out_size,1],dtype=self.precision)
        seg_idx_h=seg_idx_h+tf.cast(zeros,self.idx_precision)
        seg_idx_l=seg_idx_l+tf.cast(zeros,self.idx_precision)
        mods=mods+zeros

        #IN, OUT: (..., in_size, out_size, 1)
        ins=tf.range(0,self.in_size,dtype=self.idx_precision)
        outs=tf.range(0,self.out_size,dtype=self.idx_precision)
        IN,OUT=tf.meshgrid(ins,outs,indexing='ij')
        IN,OUT=tf.expand_dims(IN,axis=-1),tf.expand_dims(OUT,axis=-1)
        if len(self.inputs_shape)>1:
            for i in range(len(self.inputs_shape)-1):
                IN=tf.expand_dims(IN,axis=0)
                OUT=tf.expand_dims(OUT,axis=0)
        zeros=tf.zeros_like(seg_idx_l)
        indices_l=tf.concat([IN+zeros,OUT+zeros,seg_idx_l],axis=-1)
        indices_h=tf.concat([IN+zeros,OUT+zeros,seg_idx_h],axis=-1)


        """
            Draw coeff from indices
        """
        #matrix_l, matrix_h, matrix: (..., in_size, out_size)
        if self.order==0:
            matrix=tf.gather_nd(self.coeff,indices_l)
        else:
            matrix_h=tf.gather_nd(self.coeff,indices_h)
            mods=tf.squeeze(mods)
            matrix=matrix_l*(tf.cast(1.0,self.precision)-mods)+\
                   matrix_h*mods
        
        #Outputs: (..., out_size)
        outputs=tf.matmul(matrix,tf.expand_dims(inputs,axis=-1),transpose_a=True)
        return tf.squeeze(outputs)


class SegmentKanLayerV2(tf.Module):
    """
        Another implementation of SegmentKanLayer
    """
    def __init__(self,
                 in_size=300,
                 out_size=300,
                 grid_size=10,
                 delta=1e-8,
                 initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                 precision=tf.float32,
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
            #coeff: (in_size, out_size, grid_size-1)
            #grids: (in_size, out_size, grid_size)
            self.coeff=tf.Variable(self.initializer([self.in_size,self.out_size,self.grid_size-1]),
                                   dtype=self.precision,
                                   name=self.name_prefix+'_coeff',
                                   trainable=True)
            grids=tf.linspace(-1.0,1.0,self.grid_size)
            for i in range(2):
                grids=tf.expand_dims(grids,axis=0)
            grids=tf.tile(grids,[self.in_size,self.out_size,1])
            self.grids=tf.cast(grids,self.precision)
            self.is_build=True
    

    def __call__(self,inputs):
        #inputs: (..., in_size)

        if not self.is_build:
            self.build()


        """
            The forward propagation
        """
        #Map inputs within (-1, 1)
        inputs=tf.cast(inputs,self.precision)
        inputs=tf.math.tanh(tf.cast(inputs,self.precision))


        """
            Rescale each input (1, in_size) from inputs (..., in_size) to span over (-1, 1)
        """
        inputs_min=tf.math.reduce_min(inputs,axis=-1,keepdims=True)
        inputs_max=tf.math.reduce_max(inputs,axis=-1,keepdims=True)
        inputs_span=inputs_max-inputs_min
        small=tf.ones_like(inputs,dtype=self.precision)*tf.cast(self.delta,self.precision)
        denominator=tf.where(inputs_span==0.0,small,inputs_span)
        inputs=tf.cast(2.0,self.precision)*(inputs-inputs_min)/denominator-tf.cast(1.0,self.precision)
        inputs=inputs*tf.cast(1.0-self.delta,self.precision)


        """
            Build the batch matrix of segment bias 
        """
        #xs: (..., in_size, out_size, 1)
        xs=tf.expand_dims(inputs,axis=-1)+\
           tf.zeros([self.in_size,self.out_size],dtype=self.precision)
        xs=tf.expand_dims(xs,axis=-1)

        #batch_matrix: (..., in_size, out_size, grid_size-1)
        #grids: (in_size, out_size, grid_size)
        batch_matrix=tf.cast((xs>=self.grids[:,:,:-1]) & (xs<self.grids[:,:,1:]),\
                             self.precision)

        """
            Compose the weight matrix 
        """
        #matrix: (..., in_size, out_size)
        #matrix=tf.einsum('...ijk,ijk->...ij',batch_matrix,self.coeff)
        matrix=tf.math.reduce_sum((batch_matrix*self.coeff),axis=-1)

        #outputs: (..., out_size, 1)
        outputs=tf.matmul(matrix,tf.expand_dims(inputs,axis=-1),transpose_a=True)
        return tf.squeeze(outputs)



if __name__=='__main__':
    #test block
    pass
