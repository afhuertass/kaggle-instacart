
import tensorflow as tf

import numpy as np 
import input_manager as input_m

TRAIN_PATH = [ "../data/train-2.pb2" ] 
path_products = "../data/csvs/products.csv"

def main(unused_args):

    input_manager = input_m.DataInstacart( path_products  , 2 )
    num_epochs = 5
    feature, target , idds = input_manager(TRAIN_PATH , num_epochs)
    # dataset_ops =  feature , labels
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    with tf.Session() as sess:
        
        sess.run( [ init_op ] )
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess , coord=coord)

        try :
            step = 0
            while not coord.should_stop():
                t = sess.run( idds )
                print("riko-loop shape")
                #print(  np.where( t != 0 )    )
                print( t ) 
                print("riko-loop {}".format(step ) )
                step = step +1 
        except tf.errors.OutOfRangeError:
            print("fuckk - se acabo lo rico ")
            
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
