
import tensorflow as tf
import dnc as dnc 

LEN = 10 
def main(unused_args):

    # test the dnc module 
    controller_config = {
        'num_hidden' : 32 , 
        'depth' : 2 ,
        'output_size' : 50
    }
    access_config = {
        'memory_size' : 128 , 
        'num_writes' : 1 ,
        'num_reads' : 1 ,
        'w_size' : 20 ,
        
    }
    OUTPUT_SIZE = 10
    dnc1 = dnc.DNC( access_config = access_config , controller_config = controller_config ,  output_size = OUTPUT_SIZE )

   
    
    # [LEN , Batch_size, feats ]
    batch_size = 2 
    shape_sample = [ 10 , batch_size , 5 ]
    seq_len = tf.constant( 10 ,  shape=[ batch_size ]  , dtype= tf.int32 )
    rn = tf.random_normal( shape_sample )

    initial_state = dnc1.initial_state( batch_size )
    
    
    
    print("test shape")
    print( rn.shape )
    output_seq , _ = tf.nn.dynamic_rnn(
        cell = dnc1 ,
        inputs = rn ,
        time_major = True,
        initial_state = initial_state 
    )
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run( init_op )
        r = sess.run( output_seq )
        print( r.shape )
    
    """
    last_rnn = tf.gather( output_seq , int( output_seq.get_shape()[0] -1  )  )
    
    with tf.Session() as sess:
        sess.run( init_op )
        r , _ = sess.run( rr )

        l = sess.run( last_rnn  )
        print("last output")
        
        print(l.shape)
        print("shape thing")
        print( r.shape )
        
    print("todo bien")
    return ""
    """
    
if __name__ =="__main__":

    tf.app.run()
