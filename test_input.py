
import tensorflow as tf

import numpy as np 
import input_manager as input_m

TRAIN_PATH = [ "../data/train.pb2" ] 
path_products = "../data/csvs/products.csv"

def main(unused_args):

    input_manager = input_m.DataInstacart( path_products , 2 , 2 )
    
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
    print("tests - human read products ")
    
    one_hot = np.array( [  1 , 0 , 0   , 1.0 , 0  ]  , dtype = np.int32 )
    one_hot2 = np.array( [  0 , 0 , 1   , 0 , 0  ]  , dtype = np.int32 )
    one_hot = np.stack( [ one_hot , one_hot2 ] , 0 )
    print(one_hot.shape )
    input_manager.to_human_read( one_hot )

    
if __name__ == "__main__":

    tf.app.run()
