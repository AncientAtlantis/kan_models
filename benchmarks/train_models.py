
def train_models():
    #load the data set
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    data=np.load(os.path.join('..','data_sets','mnist.npz'))
    x_test,y_test,x_train,y_train=data['x_test'],data['y_test'],data['x_train'],data['y_train']
    y_test,y_train=to_categorical(y_test),to_categorical(y_train)
    x_test,x_train=tf.reshape(x_test,[-1,784]),tf.reshape(x_train,[-1,784])
    x_test,x_train=tf.cast(x_test/255,tf.float32),tf.cast(x_train/255,tf.float32)
    

    #define the model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import CategoricalAccuracy    
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    import copy

    layer_configs=copy.deepcopy(default_layer_configs)
    layer_configs['fourier']['grid_size']=20

    model=Sequential(in_size=784,\
                     n_neurons=[10],
                     layer_types=['spline'],\
                     layer_configs=layer_configs,\
                     post_ac_func=tf.nn.softmax,\
                     name_prefix='Sequential_spline')
    model.build()

    lr_schedule=ExponentialDecay(1e-3,decay_steps=200,decay_rate=0.93,staircase=True)
    optimizer=Adam(learning_rate=lr_schedule)
    model.setup(optimizer=optimizer,\
                loss=CategoricalCrossentropy(),\
                metrics=[CategoricalAccuracy()])

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
    tf.saved_model.save(model,'saved_model_fourier',signatures=signatures,options=tf.saved_model.SaveOptions(function_aliases={}))

