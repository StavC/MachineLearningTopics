import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def main():

    observations=1000
    xs=np.random.uniform(low=-10,high=10,size=(observations,1))
    zs=np.random.uniform(low=-10,high=10,size=(observations,1))

    generated_inputs=np.column_stack((xs,zs))
    noise=np.random.uniform(-1,1,(observations,1))
    generated_targets=2*xs-3*zs+5+noise
    np.savez('TF_Minimal_Example',inputs=generated_inputs,targets=generated_targets)

    training_data=np.load('TF_Minimal_Example.npz')
    input_size=2
    output_size=1

    model=tf.keras.Sequential([tf.keras.layers.Dense(output_size)])

    '''
        model=tf.keras.Sequential([tf.keras.layers.Dense(output_size,
                                                        kernerl_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                                        bias_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1])
        custom_optimizer=tf.keras.optimizers.SGD(learning_rate=0.02)
    
    
    
    '''


    model.compile(optimizer='sgd',loss='mean_squared_error')
    print(model.fit(training_data['inputs'],training_data['targets'],epochs=100,verbose=2))

    #### Extract the weights and bias

    print(model.layers[0].get_weights())
    weights=model.layers[0].get_weights()[0]
    bias=model.layers[0].get_weights()[1]

    #### Extract the outputs(make predictions)

    predicted=model.predict_on_batch(training_data['inputs']).numpy().round(1)
    print(predicted)
    targets=training_data['targets'].round(1)
    print(targets)

    plt.plot(np.squeeze(predicted),np.squeeze(targets))
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()




if __name__ == '__main__':
    main()