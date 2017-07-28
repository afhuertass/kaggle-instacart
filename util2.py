
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC util ops and modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

thr = 0.3 
def human( p , ids  ):
  # recieves the prediction and ids, and generate the output 
  #  p [ batch_size , ITEMS]
  # idds [batch_size]
  # we then the index of the
  
  for x in np.arange(0 , p.shape[0]):
    mask = p[x][:] > 0.8
    elements = np.nonzero( mask )
    idd = str(ids[x][0]) + ","
    
    resp = ""
    for element in elements[0]:
      resp += str(element) + " "
      
    if resp == "":
      resp = "None"

    
    yield (idd + resp + "\n")


           
def cost( last_output , target ):
    # last output [ batch_size , ToT ]
    # target [ batch_size , TOT ]

    # sigmoid cross entropy, because the classes arent mutially exclusive

  
  xent = tf.nn.sigmoid_cross_entropy_with_logits( logits = last_output , labels = target )
  xent_batch = tf.reduce_mean( xent , axis = 1)
  
  batch_size = tf.cast(tf.shape(last_output)[0], dtype=xent_batch.dtype)
        
  xent_total = tf.reduce_sum( xent_batch ) / batch_size
  
  print(  xent_total.shape ) 
  return  xent_total


  
def batch_invert_permutation(permutations):
  """Returns batched `tf.invert_permutation` for every row in `permutations`."""
  with tf.name_scope('batch_invert_permutation', values=[permutations]):
    unpacked = tf.unstack(permutations)
    inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
    return tf.stack(inverses)


def batch_gather(values, indices):
  """Returns batched `tf.gather` for every row in the input."""
  with tf.name_scope('batch_gather', values=[values, indices]):
    unpacked = zip(tf.unstack(values), tf.unstack(indices))
    result = [tf.gather(value, index) for value, index in unpacked]
    return tf.stack(result)


def one_hot(length, index):
  """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
  result = np.zeros(length)
  result[index] = 1
  return result
