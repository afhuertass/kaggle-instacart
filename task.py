import tensorflow as tf
from tensorflow.python.lib.io import file_io


import cPickle
import numpy as np
import sys

import dnc
import input_manager
import input_manager2 as im

import util2 as util

# 49688
OUTPUT_SIZE = 49690
BATCH_SIZE =  100
LEN = 150 


PATH_TRAIN_DATA = [ "../data/train.pb2" ]

PATH_TEST_DATA = ["../data/test_set.pb2" ]

PATH_PRODUCTS = "gs://kaggleun-instacart/data/products/products.csv"
CHECK_DIR = "../checkpoints-3"
TB_DIR = "../tensorboard"


# total train objects = 50000
n = 100 # actual number of runnings over all the training data 
NUM_ITER = 5

NUM_ITER = 5  # numero the training epochs 
NUM_ITER_TEST = 1 # para obtener las prediciones

#test delete for training
#NUM_ITER = 100
#NUM_ITER_TEST = 10
#
REP_INTERVAL = 100 
MAX_GRAD_NORM = 50
LEARN_RATE = 1e-3
MULTIPLIER = 1
reduce_learning_interval = 1000
EPSILON = 1e-4

CHECK_INTERVAL = 1000



access_config = {
        'memory_size': 128 ,
        'num_writes' : 1 ,
        'num_reads' : 2  ,
        'w_size' : 64 
    }
controller_config = {
        'num_hidden'  : 16 ,
        'depth' : 2 ,
        'output_size' : OUTPUT_SIZE
    }


def run_model2( dnc_core , initial_state  , inputs_sequence , seqlen  , output_size ):


    
    print("wtf men")
    print( input_sequence.shape )
    output_sequence , _ = tf.nn.dynamic_rnn(
        cell = dnc_core ,
        inputs = inputs_sequence ,
        sequence_length= seqlen,
        time_major = False ,
        initial_state = initial_state 
    )
    
    return output_sequence 

def train( num_epochs , rep_interval):

    
    ## create the dnc_core
    total_steps = num_epochs
    steps_test = 75000/BATCH_SIZE
    
    print("TOTAL STEPS:{}".format(total_steps ))
    #total_steps = 10 
    dnc_core = dnc.DNC( access_config = access_config , controller_config  = controller_config , output_size = OUTPUT_SIZE )
    
    initial_state = dnc_core.initial_state(BATCH_SIZE)

    #load the data
    #input_data = input_manager.DataInstacart( PATH_PRODUCTS, BATCH_SIZE  )
    #input_data_test = input_manager.DataInstacart( PATH_PRODUCTS ,BATCH_SIZE )

    input_data_train = im.InputManager( BATCH_SIZE , PATH_TRAIN_DATA[0] )
    input_data_test = im.InputManager( BATCH_SIZE , PATH_TEST_DATA[0] , 1 )

    iterator_train = input_data_train.data.make_initializable_iterator()
    iterator_test = input_data_test.data.make_initializable_iterator()

    
    #input_tensors = input_data(PATH_TRAIN_DATA , num_epochs ) # training input 
    #input_tensors_test = input_data_test(PATH_TEST_DATA , 50 )

    input_tensors = iterator_train.get_next()
    input_tensors_test = iterator_test.get_next()
    
    
    output_sequence = run_model2( dnc_core , initial_state , input_tensors[0] , input_tensors[3] , OUTPUT_SIZE  )

    output_sequence_test = run_model2(  dnc_core , initial_state , input_tensors_test[0] , input_tensors_test[3] , OUTPUT_SIZE )
    # last output from the recurrent neural network 
    last_rnn = tf.gather( output_sequence , int( output_sequence.get_shape()[0] - 1  )  )

    last_rnn_test = tf.gather( output_sequence_test , int( output_sequence_test.get_shape()[0] - 1  )  )


    ### add to collection to recover for later testing
    
    tf.add_to_collection('outputs_test', last_rnn_test )
    tf.add_to_collection('outputs_test', input_tensors_test[2] )
    
    train_loss = util.cost(  last_rnn , input_tensors[1] )
    print("sheips")
    print( last_rnn.shape )
    return 
    #eval_loss = input_data.cost_f1( last_rnn , input_tensors[1] )
    
    
    tf.summary.scalar( 'loss' , train_loss  )
    
    trainable_variables = tf.trainable_variables()

    grads , _ = tf.clip_by_global_norm(
        tf.gradients( train_loss , trainable_variables) , MAX_GRAD_NORM
    )

    learning_rate = tf.get_variable(
        "learning_rate" , shape = [],
        dtype = tf.float32 , initializer = tf.constant_initializer(LEARN_RATE) ,
        trainable = False 
    )
    
    reduce_learning_rate = learning_rate.assign( learning_rate*MULTIPLIER  )
    
    global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
    )

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate , epsilon = EPSILON
    )

    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables) , global_step = global_step
    )


    
    saver = tf.train.Saver( )
    tf.summary.scalar( 'loss' , train_loss  )

    merged_op = tf.summary.merge_all()

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




    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    
    with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = CHECK_DIR , config = config ) as sess:

        writer = tf.summary.FileWriter( TB_DIR , sess.graph )
       
        
        start_iteration = sess.run(global_step)
        # initialize iterator
        sess.run( iterator_train.initializer )
        sess.run( iterator_test.initializer )
        
        total_loss = 0

        print("start:{}".format( start_iteration ) )
        
        for train_iteration in xrange(start_iteration , total_steps ):

            #t =  sess.run( input_tensors[0] ) # feats
            # training step
            _ , loss = sess.run( [ train_step , train_loss] )
            
            if train_iteration % 100 == 0 :
                summary  = sess.run( merged_op  )
                writer.add_summary(summary , train_iteration )
            

            if ( train_iteration  + 1 )% reduce_learning_interval == 0:
                sess.run( reduce_learning_rate )
                print("reducing learning rate")
                
            if train_iteration % 100 == 0 :
                
                print( "loss:{}".format(loss)  )
               
            print( "step-training:{}/{}".format( train_iteration, total_steps ) )
            

       
        i = 1
        string_to_file="order_id,products\n"
        
        try:
            while True :
                prediction , idd = sess.run( [ last_rnn_test , input_tensors_test[2]  ] )

                result = util.human( prediction , idd )
                for r in result:
                    string_to_file += r
            
            
                print( "step test, samples:{}/{}".format( i*100 , 75000 ) )
                i = i +1 

        except tf.errors.OutOfRangeError:
            print("Queue dead ")
            
        test_file = "sub.txt"
        test = open(test_file , 'w')
        test.write( string_to_file )
        test.close()
    
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    

   
    
def test( test_file ):
    # restore an generate test file
    string_to_file = "order_id,products\n"
    modelfile = "../checkpoints-2/model.ckpt-40000.meta"

    all_ids = [] 
    with tf.Session() as sess:
        
        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )
        g = tf.get_default_graph()

        saver = tf.train.import_meta_graph( modelfile )
        saver.restore( sess , tf.train.latest_checkpoint("../checkpoints-2/") )


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord= coord)

        tensors = tf.get_collection('outputs_test')
        print( tensors )
        
        recuperado_last_rnn = tensors[0]
        recuperado_idd = tensors[1]
        steps = 1000/50

        i = 0
       
        while not i >= 20 :
            prediction , idd = sess.run( [ recuperado_last_rnn , recuperado_idd ] )
            all_ids.append( idd )
            
            print( idd[0][0] )
            result = util.human( prediction , idd )
            for r in result:
                string_to_file += r
                
                
            print( "step test:{}/{}".format(i , steps ) )
            i = i +1 
     


    test = open(test_file , 'w')
    test.write( string_to_file )
    test.close()

    test2 = open("./ids.txt" , 'w')
    cc = ""
    for idd in all_ids:

        for x in np.arange( 0 , idd.shape[0]) :
            cc += str( idd[x][0 ]) + "\n"

    test2.write( cc )
    test2.close()
    
def main( unuser_args):

    train( 10 , REP_INTERVAL)

    #test( "./sub-32000.txt")
    
    print("riko -train ")

if __name__=="__main__":

    tf.app.run()
