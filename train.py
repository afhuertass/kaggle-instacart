    
import tensorflow as tf
import numpy as np

import dnc
import input_manager

OUTPUT_SIZE = 49690
BATCH_SIZE = 2
PATH_PRODUCTS = "../data/csvs/products.csv"
PATH_TRAIN_DATA = [ "../data/train-2.pb2" ]
PATH_TEST_DATA = ["../data/train-2.pb2" ]

NUM_ITER = 100
REP_INTERVAL = 100

MAX_GRAD_NORM = 50
LEARN_RATE = 1e-4
EPSILON = 1e-6

CHECK_DIR = "../checkpoints"
CHECK_INTERVAL = 100

access_config = {
        'memory_size': 128 ,
        'num_writes' : 1 ,
        'num_reads' : 2  ,
        'w_size' : 64 
    }
controller_config = {
        'num_hidden'  : 8 ,
        'depth' : 2 ,
        'output_size' : OUTPUT_SIZE
    }

def run_model( inputs_sequence , output_size ):
    # build the model
  
    
    dnc_core = dnc.DNC( access_config = access_config , controller_config  = controller_config , output_size = output_size )

    initial_state = dnc_core.initial_state(BATCH_SIZE)

    output_sequence , _ = tf.nn.dynamic_rnn(
        cell = dnc_core ,
        inputs = inputs_sequence ,
        time_major = True ,
        initial_state = initial_state 
    )
    return output_sequence

def run_model2( dnc_core , initial_state  , inputs_sequence  , output_size ):


    output_sequence , _ = tf.nn.dynamic_rnn(
        cell = dnc_core ,
        inputs = inputs_sequence ,
        time_major = True ,
        initial_state = initial_state 
    )

    return output_sequence 

def train( num_epochs , rep_interval):



    
    ## create the dnc_core 
    dnc_core = dnc.DNC( access_config = access_config , controller_config  = controller_config , output_size = OUTPUT_SIZE )
    
    initial_state = dnc_core.initial_state(BATCH_SIZE)

    #load the data
    input_data = input_manager.DataInstacart( PATH_PRODUCTS, BATCH_SIZE  )
    input_tensors = input_data(PATH_TRAIN_DATA , num_epochs )

    # load the test data
    input_tensors_test = input_data(PATH_TEST_DATA , 1 ) # una sola pasada 
    
    output_sequence = run_model2( dnc_core , initial_state , input_tensors[0] , OUTPUT_SIZE  )

    output_sequence_test = run_model2(  dnc_core , initial_state , input_tensors_test[0] , OUTPUT_SIZE )
    # last output from the recurrent neural network 
    last_rnn = tf.gather( output_sequence , int( output_sequence.get_shape()[0] - 1  )  )

    last_rnn_test = tf.gather( output_sequence_test , int( output_sequence_test.get_shape()[0] - 1  )  )
    
    print("wash going on two")
    print(last_rnn.shape)
    print( input_tensors[1])
    train_loss = input_data.cost(  last_rnn , input_tensors[1] )


    trainable_variables = tf.trainable_variables()

    grads , _ = tf.clip_by_global_norm(
        tf.gradients( train_loss , trainable_variables) , MAX_GRAD_NORM
    )

    global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
    )

    optimizer = tf.train.RMSPropOptimizer(
        LEARN_RATE , epsilon = EPSILON
    )

    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables) , global_step = global_step
    )

    saver = tf.train.Saver()

    if CHECK_INTERVAL > 0:

        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir = CHECK_DIR ,
                save_steps = CHECK_INTERVAL ,
                saver = saver 
            )
        ]
    else:
        hooks = []

   
    
    with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = CHECK_DIR ) as sess:

       
        start_iteration = sess.run(global_step)
        total_loss = 0

            
        for train_iteration in xrange(start_iteration , num_epochs):

            t =  sess.run( input_tensors[0] ) # feats 
            
            _ , loss = sess.run( [ train_step , train_loss] )
            print(loss )

            print( "run test" )
            last_rnn_test_run = sess.run( last_rnn_test  )
            
            print( last_rnn_test_run)
            print( "prediction ")
            human = input_data.to_human_read( last_rnn_test_run )
            for e in human:
                print(e)
                #total_loss  += loss
                

    ## provide predictions
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train_start_queue_runners( sess = sess , coord = coord )
        try:
            step = 0
            while not coord.should_stop():
                print( "step:{}".format( step ) )
                # retrieve prediction , y el id 
                prediction , idd  = sess.run( [last_rnn_test , output_sequence_test[2] ] )
                human = input_data.to_human_read( prediction  )
                print(  "iddd")
                print( idd ) 
                for e in human :
                    step += 1 
                    print( e )
        except tf.errors.OutOfRangeError :
            print("Exhausted Queue ")
        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
            
        
        
        
def main( unuser_args):

    train(NUM_ITER , REP_INTERVAL)
    
    print("riko -train ")

if __name__=="__main__":

    tf.app.run()
