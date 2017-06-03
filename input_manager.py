
import multiprocessing

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

import sonnet as snt

# this is for the onehot encoding vector

TOTAL_ITEMS = 49690 # total podructs
LEN = 100  # 
def parse_examples( examples ):
        # parse the data from the file, to an apropiated format
    
    feature_map = {
        'target': tf.VarLenFeature(
            dtype = tf.int64
        ) ,
        'feature': tf.VarLenFeature(
        dtype = tf.int64
        )
    }
    parsed = tf.parse_example(examples , feature_map )
    
    """
    so far here we have feature = [ n_inputs ] , 
    target = [ n_targets ] ,  targets should be a one hot encoding
    
    """
    print("shape target")
   

    sparse_target_indices = tf.reshape( parsed['target'].values , shape=(-1 , ) )
    print("shape target indiciiiii")
    print(sparse_target_indices.shape)
   
    
    target = tf.one_hot( sparse_target_indices , TOTAL_ITEMS , on_value = 1.0 ,
                         off_value = 0.   )

    print(target.shape )
    
    #target = tf.one_hot( parsed['target'] , TOTAL_ITEMS , on_value = 1.0 ,
                         #off_value = 0. )
    # target [ n_target , TOTAL_ITEMS ]
    
    print("shape target reduced")
    target = tf.reduce_sum( target , 0   )
    
    target = tf.reshape( target , [ -1 ,TOTAL_ITEMS ] )
    print(target.shape) 
    print("targets good")
    
    sparse_feature_indices = tf.reshape(  parsed['feature'].values[-LEN:] , shape=(-1 , ) )
    
    #sparse_feature_indices = tf.to_int64( sparse_feature_indices )
    print("sparse features")
    features = tf.one_hot( sparse_feature_indices , TOTAL_ITEMS , on_value = 1.0 ,
                           off_value = 0.0 ,  )
    
    print( features.shape )
    features = tf.reshape( features , shape = ( LEN  ,TOTAL_ITEMS )  )
    
    print("one hot re hot ")
        
    # target [ TOTAL_ITEMS]
    # features [ n_inputs , TOTAL_ITEMS ] total 

    print("shapes FINAL")
    print( features.shape )
    print( target.shape )
    
    return features , target
    

class DataInstacart(snt.AbstractModule):

    def __init__(self , batch_size , num_epochs , name = 'data_m'):


        super(DataInstacart , self ).__init__(name)

        self.batch_size = batch_size

        self.num_epochs = num_epochs

        self.shape_sample = [ LEN , batch_size , TOTAL_ITEMS  ]
        self.shape_target = [ 1 , batch_size , TOTAL_ITEMS ]
            
    def _build(self, data_files):
        # recieves the data files where the .tfrecord lies.
        # so it build the queue to read from the files and retrieve the data tensors
        
       
        # build the shit
        print(data_files)
        thread_count = multiprocessing.cpu_count()
        min_after_dequeue = 1000
        queue_size_multiplier = thread_count + 3
        capacity = self.batch_size*2
        
        filename_queue = tf.train.string_input_producer(   data_files    )
        
        _ , encoded_examples = tf.TFRecordReader(
            options = tf.python_io.TFRecordOptions  (
                compression_type= TFRecordCompressionType.GZIP
            )
        ).read_up_to( filename_queue , self.batch_size)


        #features, target = parse_examples( encoded_examples )
        feature, target =  parse_examples( encoded_examples )
        # define certain capacity
        

        print("shapes train batchs")
        print( feature.shape )
        print( target.shape  )
        
        # of change to a shuffle batch 
        result = tf.train.batch(
            [feature,target] , batch_size = self.batch_size  ,
            capacity = capacity ,
            allow_smaller_final_batch=True ,
            enqueue_many = False ,
           
        )

        result[0] = tf.reshape( result[0] , self.shape_sample )
        result[1] = tf.reshape(  result[1] , self.shape_target )
        #print( result )
        # feature [ LEN , batch_size , TOTAL ]
        # target  [ batch_size ,  TOTAL ]
        return result  
