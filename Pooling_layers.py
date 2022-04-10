import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras import backend as K

class MaxPooling2DMod(layers.Layer):
  """ class for tf.nn.max_pool_with_argmax func.
  """
  def __init__(self,               
               ksize=(2, 2),
               strides=(2, 2),
               padding='SAME',
               **kwargs):
    super(MaxPooling2DMod, self).__init__()
    self.padding = padding
    self.ksize = ksize
    self.strides = strides

  # overwrite get_config method to correct save model (model.save()) in h5
  # when __init__() have def. parameters
  
  def call(self, inputs):
    ksize = [1, self.ksize[0], self.ksize[1], 1]
    strides = [1, self.strides[0], self.strides[1], 1]
    max_pooling_out, argmax = tf.nn.max_pool_with_argmax(inputs, 
                                                        ksize,
                                                        strides, 
                                                        padding=self.padding, 
                                                        include_batch_in_index=True)

    return [max_pooling_out, argmax]


class MaxUnpooling2DMod(layers.Layer):

  def __init__(self,               
               **kwargs):
    super(MaxUnpooling2DMod, self).__init__()

  def max_unpool(self, inputs, pooling_indices, output_shape=None):
    # https://github.com/fregu856/segmentation.git  .... MIT license

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = inputs.get_shape()

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices*batch_range
    y = pooling_indices//(output_shape[2]*output_shape[3])
    x = (pooling_indices//output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices*feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])

    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

  def call(self, inputs, pooling_indices, output_shape):
    unpool = self.max_unpool(inputs, pooling_indices, output_shape)
    return unpool

def upsample(x, out_shape, style='tf'):
    """
    This func. take output from tf.nn.max_pool_with_argmax (maxpool, argmax_index)
    and original shape of tensor and make upsample tensor.

    input - output from tf.nn.max_pool_with_argmax (maxpool value)
    armax - second output from tf.nn.max_pool_with_argmax 
    out_shape - shape of orignal tensor

    # """
    input, argmax = x
    input_shape = input.shape
    batch_size = input_shape[0]
    out_shape = (batch_size, out_shape[1], out_shape[2], input_shape[3])
    
    if style == 'tf':
        # shift idx to correct place
        lins = tf.linspace(0, batch_size-1, batch_size)
        
        shift = lins*out_shape[1]*out_shape[2]*out_shape[3]
        shift = tf.reshape(shift, (batch_size, 1, 1, 1))
        shift = tf.cast(shift, dtype="int64")

        # correct idx
        argmax_correct = argmax + shift

        #helper = layers.Flatten()
        #argmax_correct = helper(argmax_correct)

        argmax_correct = tf.reshape(argmax_correct, [1, -1])

        new_shape = argmax_correct.shape[0]*argmax_correct.shape[1]

        argmax_correct = tf.reshape(argmax_correct, (new_shape, 1))
        argmax_correct = tf.cast(argmax_correct, dtype="int32")

        #input_f = helper(input)
        input_f = tf.reshape(input, [1, -1])

        input_f = tf.reshape(input_f, [new_shape])

        # new upsample tensor
        shape = tf.constant([batch_size*out_shape[1]*out_shape[2]*out_shape[3]])
        output = tf.scatter_nd(argmax_correct, input_f, shape)

        return tf.reshape(output, out_shape)

    elif style == 'numpy':
        input_shape = input.get_shape()
        input_n = input.numpy()
        argmax_n = argmax.numpy()

        batch_size = out_shape[0]

        # correction of argmax due to batch size
        lins = np.linspace(0, batch_size-1, batch_size)
        shift = lins * np.prod(out_shape[1:4]) 
        shift = np.reshape(shift, (input_shape[0], 1, 1, 1))

        argmax_correct = argmax_n + shift
        argmax_correct = argmax_correct.flatten().astype('int32')

        # numpy for output
        output_n = np.zeros(out_shape)

        # replace zero for input[correct argmax] 
        output_n = output_n.flatten()
        output_n[argmax_correct] = input.numpy().flatten()

        # return tensor
        return tf.constant(np.reshape(output_n, out_shape))

class UpSampling2DwithArgmax(layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(UpSampling2DwithArgmax, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape):
        max_pool_input, argmax = inputs[0], inputs[1]
        argmax = tf.cast(argmax, 'int32')
        max_pool_input_shape = tf.shape(max_pool_input, out_type='int32')

        output_shape = tf.shape(output_shape, 'int32')
        output_shape = (max_pool_input_shape[0], output_shape[1],
                            output_shape[2], max_pool_input_shape[3])
        
        shape = [tf.math.reduce_prod(output_shape)]
        input_f = tf.reshape(max_pool_input, [1,-1])
        argmax_correct = tf.expand_dims(tf.reshape(argmax, [1,-1]), axis=2)

        output = tf.scatter_nd(argmax_correct, input_f, shape)   
        output_shape = [-1,
                        output_shape[1],
                        output_shape[2],
                        output_shape[3]]

        return tf.reshape(output, output_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'size': self.size
        })
        return config

class MaxUnpooling2D22(layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D22, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])
        else:
            output_shape = tf.shape(output_shape, 'int32')

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                            K.flatten(updates),
                            [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )



if __name__ == '__main__':

  x1 = tf.random.uniform([50,4,4,3],maxval=100,dtype='float32',seed=2)

  print('Original tensor is \n{}'.format(x1))

  max_poll_mod = MaxPooling2DMod()
  out, idx = max_poll_mod(x1)

  print('Tensor after MaxPooling2DMod (modify) \n{}'.format(out))

  unpool = UpSampling2DMod()([out, idx], x1.shape)

  print('Tensor after UpSampling2D1 \n{}'.format(unpool))

  # print('Tensor after MaxPooling2D layer (tf imple) \n{}'.format(max_poll(x1)))

  # print('Argmax from MaxPooling2DMod \n{}'.format(idx))

  # tic = time.perf_counter()
  # unpoll_mod = MaxUnpooling2DMod()
  # unpoll_ten1 = unpoll_mod(out, idx, x1.shape)
  # toc = time.perf_counter()
  # print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

  # print('Tensor after MaxUnpooling2DMod \n{}'.format(unpoll_ten1))

  # print(unpoll_ten==unpoll_ten1)


  # tic = time.perf_counter()
  # unpoll_mod = layers.UpSampling2D((2,2))
  # unpoll_ten = unpoll_mod(out)
  # toc = time.perf_counter()
  # print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
  # print('Tensor after MaxUnpooling2DMod \n{}'.format(unpoll_ten))
