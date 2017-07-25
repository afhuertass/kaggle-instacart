

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
        
        self.data = self.dataset.batch( batch_size )

        

    def _parse_training(self , example  ):
        
        feature_map = {
            
            'target': tf.VarLenFeature(
                dtype = tf.int64 
            ) ,
            'feature' : tf.VarLenFeature(
                dtype = tf.int64
            ) ,
            'ids' : tf.VarLenFeature(
                dtype = tf.int64
            )
        }
        print( example ) 
        #parsed = tf.parse_example( example , feature_map )

        parsed = tf.parse_single_example( example , feature_map  )

        print( parsed['feature'] )
        #  idd of the sample
        idd = tf.reshape( parsed['ids'].values[:1] , shape =[1])
        #targets 
        sparse_target_indices = tf.reshape( parsed['target'].values[:], shape= [-1] )

        target = tf.one_hot( sparse_target_indices , TOTAL_ITEMS, on_value=1.0 , off_value = 0.0 )

        target = tf.reduce_sum(target , 0 )
        target = tf.reshape( target , [ -1 ,TOTAL_ITEMS ] )
        
        sparse_feature_indices = tf.reshape( parsed['feature'].values[:LEN] , shape= [LEN])

        seqlen = tf.count_nonzero(  sparse_feature_indices   )
        
        features = tf.reshape( sparse_feature_indices , shape = [ LEN , 1  ]  )
        features = tf.cast( features , tf.float32 )

        print("shape features")
       
        
        return features , target , idd , seqlen 
        
    
        
        
    
