import tensorflow as tf


class SegmentKanLayer(tf.Module):
    """
    The
    """
    def __init__(self,
                 in_size,
                 out_size,
                 num_grids,
                 delta=1e-8,
                 initializer,
                 name_prefix='SegmentKanLayer'):
        super().__init__(name=name_prefix)
        self.in_size=in_size
        self.out_size=out_size
        self.num_grids=num_grids
        self.delta=delta
        self.initializer=initializer
        self.name_prefix=name_prefix    
        self.is_build=False
    
    def build(self):
        if not self.is_build:
            
            self.is_build=True


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
