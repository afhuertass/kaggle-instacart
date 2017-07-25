

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
class InputManager( snt.AbstractModule):


    def __init__( self , batch_size , data_path , name="data-pipline" ):
        
        
        self.batch_size = batch_size
        

        self.shape_sample = [ LEN , -1 , 1  ]
        self.shape_target = [-1 , TOTAL_ITEMS ]
        self.shape_ids = [-1 , 1 ]

        self.data = None

        self.load_data( data_path )

        #load de data from a tfrecord 
        self.dataset = TFRecordDataset( data_path , "GZIP"  )
        
   

    def _build( ):
        

    
