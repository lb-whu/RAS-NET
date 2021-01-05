import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

### RES-unet layers
def layer_down(input, k, is_train,reuse=False, keep_prob= 1, is_training=True, name='layer_down'):
  with tf.variable_scope(name, reuse=reuse):

    #conv1
#    input = tf.squeeze(input,1)
    weights_1 = _weights("weights_1",is_train=is_train,
      shape=[3, 3, input.get_shape()[3], k])
    biases_1 = _biases("biases_1", is_train=is_train,shape =[k])
    result_conv_1 = tf.nn.conv2d(input, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'),is_train=is_train, is_training = is_training, norm='batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",is_train=is_train,
      shape=[3, 3, result_relu_1.get_shape()[3], k])
    biases_2 = _biases("biases_2",is_train=is_train, shape =[k])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_train=is_train,is_training = is_training, norm='batch',name = 'conv2')
                  
    #resnet cov
    weights_11 = _weights("weights_11",is_train=is_train,
      shape=[1, 1, input.get_shape()[3], k])
    res_input = tf.nn.conv2d(input, weights_11,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_11')
                                                                                                                         
    resnormed = tf.add(normalized2,res_input)
    
    result_relu_3 = tf.nn.relu(resnormed, name='relu_3')
    
    # maxpool
    result_maxpool = tf.nn.max_pool(value=result_relu_3, ksize=[1, 2, 2, 1],
				                  strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

		# dropout
#    result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=keep_prob)
    return result_relu_3,result_maxpool

def layer_mid(input, k_down,k_up, is_train,reuse=False, keep_prob= 1, is_training=True, name='layer_mid'):
  with tf.variable_scope(name, reuse=reuse):
    result = tf.image.resize_bilinear(input,[input.get_shape()[1]*2,
                            input.get_shape()[2]*2])
    weights_3 = _weights("weights_3",is_train=is_train,
      shape=[1, 1, result.get_shape()[3], k_up])
    biases_3 = _biases("biases_3",is_train=is_train, shape =[k_up])
    result_conv_3 = tf.nn.conv2d(result, weights_3,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
    result_relu = tf.nn.relu(tf.nn.bias_add(result_conv_3, biases_3, name='add_bias'), name='relu_3')
		# dropout
    result_dropout = tf.nn.dropout(x=result_relu, keep_prob=keep_prob)
    return result_dropout

def layer_up(result_from_contract_layer,result_from_upsampling, k_down,k_up,is_train,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  with tf.variable_scope(name, reuse=reuse):
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    
    
    #conv1
    print('result_merge:',result_merge.shape)
    weights_1 = _weights("weights_1",is_train=is_train,
      shape=[3, 3, result_merge.get_shape()[3], k_down])
    biases_1 = _biases("biases_1",is_train=is_train, shape =[k_down])
    result_conv_1 = tf.nn.conv2d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'),is_train=is_train, is_training = is_training, norm='batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",is_train=is_train,
      shape=[3, 3, result_relu_1.get_shape()[3], k_down])
    biases_2 = _biases("biases_2",is_train=is_train, shape =[k_down])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_train=is_train,is_training = is_training, norm='batch',name = 'conv2')
    
    #resnet cov
    weights_11 = _weights("weights_res",is_train=is_train,
      shape=[1, 1, result_merge.get_shape()[3], k_down])
    res_input = tf.nn.conv2d(result_merge, weights_11,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_res')                                                                                   
    resnormed = tf.add(normalized2,res_input)
    
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
    result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)
    
    # unsample
    result = tf.cast(tf.image.resize_bilinear(tf.cast(result_dropout,tf.float32),[result_dropout.get_shape()[1]*2,result_dropout.get_shape()[2]*2]),tf.float32)
    
    weights_3 = _weights("weights_3",is_train=is_train,
      shape=[1, 1, result.get_shape()[3], k_up])
    biases_3 = _biases("biases_3",is_train=is_train,shape = [k_up])
    result_conv_3 = tf.nn.conv2d(result, weights_3,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
    result_relu = tf.nn.relu(tf.nn.bias_add(result_conv_3, biases_3, name='add_bias'), name='relu_3')
		# dropout
#    result_dropout = tf.nn.dropout(x=result_relu, keep_prob=keep_prob)

    return result_relu

def layer_last(result_from_contract_layer,result_from_upsampling, k_down,num_class,is_train,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    #conv1
    weights_1 = _weights("weights_1",is_train=is_train,
      shape=[3, 3, result_merge.get_shape()[3], k_down])
    biases_1 = _biases("biases_1", is_train=is_train,shape =[k_down])
    result_conv_1 = tf.nn.conv2d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",is_train=is_train,
      shape=[3, 3, result_relu_1.get_shape()[3], k_down])
    biases_2 = _biases("biases_2",is_train=is_train, shape =[k_down])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), name='relu_2')
    
    #last conv
    weights_3 = _weights("weights_3",is_train=is_train,
      shape=[1, 1, k_down, num_class])
    biases_3 = _biases("biases_3", is_train=is_train,shape =[num_class])
    result_conv_3 = tf.nn.conv2d(result_relu_2, weights_3,
				                              strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
    prediction = tf.nn.bias_add(result_conv_3, biases_3, name='add_bias')
    return prediction


def copy_and_crop_and_merge(result_from_contract_layer,result_from_upsampling):
  return tf.concat(values=[result_from_contract_layer, result_from_upsampling], axis=-1)
  
def _weights(name, shape, is_train,mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,trainable=is_train,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, is_train,constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,trainable=is_train,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_train,is_training, norm='instance',name = 'conv'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_train = is_train,is_training = is_training,bname = name)
  else:
    return input

def _batch_norm(input, is_train,is_training,bname = 'conv'):
  """ Batch Normalization
  """
  with tf.variable_scope(bname+"_batch_norm"):
    '''
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)
    '''
    return tf.layers.batch_normalization(input,
             #                           decay=0.9,
                                        scale=True,
             #                           updates_collections=None,
                                        trainable = is_train,
                                        training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
