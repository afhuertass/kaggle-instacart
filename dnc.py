import collections


import tensorflow as tf
import controller as contr
import sonnet as snt


DNCState = collections.namedtuple('DNCState' , ('access_output','access_state' , 'controller_state' ) )


class DNC( snt.RNNCore ):

    def __init__(self , access_config , controller_config , name = "my-dnc" ):


        super(DNC , self).__init__(name)
        # create the controller
        self._controller = contr.RnnInstacart(**controller_config )
        
    
            
        return

    def _build(self , inputs , prev_state ):

        # inputs are [ LEN , batch_size , TOT]
        # prev_state for the network
        # prev_state.controller_state # is needed
          
        batch_flatten = snt.BatchFlatten()
        """
        prev_controller_state = prev_state.controller_state
        prev_access_output = prev_state.access_output 

        
        controller_input = tf.concat([batch_flatten( inputs ) , batch_flatten(  prev_access_output    )] , 1 )
        """
        controller_output , controller_state = self._controller( inputs   )


        ## TODO ADD LINEAR LAYER 

        return controller_output , DNCState(
            controller_state = controller_state,
            access_state = tf.random_normal([1]) ,
            access_output =  tf.random_normal([1]) ,

        ) 
    def initial_state(self , batch_size , dtype = tf.float32  ):

        return DNCState(
            controller_state = self._controller.initial_state(batch_size) ,
            access_sate = 'TODO' ,
            access_output = 'TODO'
        )

    
