
import tensorflow as tf

import numpy as np 
import input_manager as input_m

TRAIN_PATH = [ "./train.pb2" ] 


def main(unused_args):

    input_manager = input_m.DataInstacart( 2 , 2 )
    
    feature, target = input_manager(TRAIN_PATH)
    # dataset_ops =  feature , labels
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    with tf.Session() as sess:
        
        sess.run( [ init_op ] )
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess , coord=coord)

        try :
            step = 0 
            while not coord.should_stop():
                t = sess.run( target )
                print("riko-loop shape")
                print(  t.shape   )
                print("riko-loop")
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            print("fuckk")
            
        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
            
    print("riko")
if __name__ == "__main__":

    tf.app.run()
