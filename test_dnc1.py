
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
        'none' : 'none' , 
    }
    dnc1 = dnc.DNC( access_config = access_config , controller_config = controller_config )

    initial_state = dnc1.initial_state() # check if method is
    
    # [LEN , Batch_size, feats ]
    batch_size = 2 
    shape_sample = [ 10 , batch_size , 5 ]
    seq_len = tf.constant( 10 ,  shape=[ batch_size ]  , dtype= tf.int32 )
    rn = tf.random_normal( shape_sample )

    rr = dnc1( rn , None ) # model build
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    
    
    last_rnn = tf.gather( rr[0] , int( rn[0].get_shape()[0] -1  )  )
    
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

if __name__ =="__main__":

    tf.app.run()
