

import tensorflow as tf

from tensorflow.contrib.data import Dataset , Iterator
from tensorflow.contrib.data import TFRecordDataset 

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor 

import sonnet as snt
import numpy as np
import cPickle as pickle 



# attempt to create a new pipeline
# using the features introduced on version 1.2 of tensorflow
LEN = 150
TOTAL_ITEMS = 49690
class InputManager():


    def __init__( self , batch_size , data_path ,  repeat = None ,name="data-pipline" ):
        
        
        self.batch_size = batch_size
        

        self.shape_sample = [ LEN , -1 , 1  ]
        self.shape_target = [-1 , TOTAL_ITEMS ]
        self.shape_ids = [-1 , 1 ]

        self.data = None

        

        #load de data from a tfrecord 
        self.dataset = TFRecordDataset( data_path , "GZIP"  )

        self.dataset = self.dataset.map( self._parse_training )
        self.dataset = self.dataset.repeat(repeat)
        self.dataset = self.dataset.shuffle( batch_size*10 )
        self.data = self.dataset.batch( batch_size )

        

    def _parse_training(self , example  ):
        
        feature_map = {
            
            'target': tf.FixedLenFeature( shape = () , 
                dtype = tf.int64 
            ) ,
            'feature' : tf.FixedLenFeature( shape = [150] , 
                dtype = tf.int64
            ) ,
            'ids' : tf.FixedLenFeature( shape = [] , 
                dtype = tf.int64
            )
        }
        print( example ) 
        #parsed = tf.parse_example( example , feature_map )

        parsed = tf.parse_single_example( example , feature_map  )
        print(  parsed['feature'].shape ) 

        features = tf.reshape( parsed['feature'] , shape =[ 150 , 1 ] )
        features = tf.cast( features , tf.float32 )
        target = tf.reshape ( parsed['target'] , shape = [] )
        target = tf.cast( target , tf.float32 )
        
        idd = tf.reshape( parsed['ids'] , shape = [] ) 
        seqlen = tf.count_nonzero( features )
    
        
        print("shape features")
        print( features.shape)
        print( target.shape )
        
        
        return features , target , idd , seqlen 
        
    
        
        
    
