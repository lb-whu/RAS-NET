import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## Layers: follow the naming convention used in the original paper
### Generator layers
def ladain(content, mean1,var1,mean2,var2, epsilon=1e-5):
    axes = [1,2]
    c_mean, c_var = tf.nn.moments(mean1, axes=axes, keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    s_mean, s_var = tf.nn.moments(mean2, axes=axes, keep_dims=True)
    s_std = tf.sqrt(s_var + epsilon)
#    s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
    
#    c_mean_f = tf.zeros_like(content)
#    c_var_f = tf.zeros_like(content)
#    s_mean_f = tf.zeros_like(style)
#    s_var_f = tf.zeros_like(style)
#    for k in range(0,8):
#      tempy = tf.expand_dims(segy[:,:,:,k],3)
#      tempz = tf.expand_dims(segz[:,:,:,k],3)
#      c_mean = tf.reduce_sum(content*tempy,axis=axes,keep_dims=True)/(tf.reduce_sum(tempy,axis=axes,keep_dims=True)+1)
#      c_var = tf.reduce_sum(tf.square(content*tempy-c_mean*tempy),axis=axes,keep_dims=True)/(tf.reduce_sum(tempy,axis=axes,keep_dims=True)+1)
#      c_mean_f = c_mean_f + c_mean * tempy
#      c_var_f = c_var_f + c_var * tempy
      
#      s_mean = tf.reduce_sum(style*tempz,axis=axes,keep_dims=True)/(tf.reduce_sum(tempz,axis=axes,keep_dims=True)+1)
#      s_var = tf.reduce_sum(tf.square(style*tempz-s_mean*tempz),axis=axes,keep_dims=True)/(tf.reduce_sum(tempz,axis=axes,keep_dims=True)+1)
#      s_mean_f = s_mean_f + s_mean * tempz
#      s_var_f = s_var_f + s_var * tempz

#    c_std, s_std = tf.sqrt(c_var_f + epsilon), tf.sqrt(s_var_f + epsilon)
#    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)
#    return s_std * (content - c_mean_f) / c_std + s_mean_f
    return s_std * (content - mean1) / c_std + mean2
#    return content
def c7s1_k(input, k, is_train,reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",is_train = is_train,
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training = is_training,norm= norm,is_train = is_train)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output
def attention(input,k, reuse=False, keep_prob= 1, is_train = True,is_training=True, norm='instance',name='layer_c'):

  with tf.variable_scope(name, reuse=reuse):

    w_1 = tf.reduce_mean(tf.reduce_mean(input, axis=1, keep_dims=True),axis=2, keep_dims=True)
    weights_11 = _weights("weights_11",is_train = is_train,shape=[1, 1, input.get_shape()[3],input.get_shape()[3]])
    w_1 = tf.nn.conv2d(w_1, weights_11,strides=[1, 1, 1, 1], padding='SAME', name='conv_11')
    
    w_2 = tf.reduce_max(tf.reduce_max(input, axis=1, keep_dims=True),axis=2, keep_dims=True)                                                                               
    weights_12 = _weights("weights_12",is_train = is_train,shape=[1, 1, input.get_shape()[3],input.get_shape()[3]])
    w_2 = tf.nn.conv2d(w_2, weights_12,strides=[1, 1, 1, 1], padding='SAME', name='conv_12')    
    w = w_1+w_2
    
    w_norm = _norm(w, is_training = is_training,norm= norm,is_train = is_train,name = 'n1')
    
    w = tf.nn.sigmoid(w_norm,name = 'sig')
    result = tf.multiply(input, w) 
    
    weights_2 = _weights("weights_2",is_train = is_train,
      shape=[1, 1, result.get_shape()[3], k])
    biases_2 = _biases("biases_2",is_train = is_train, shape = [k])
    result_conv_2 = tf.nn.conv2d(result, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training = is_training,norm= norm,is_train = is_train,name = 'n2')   
    result_relu_2 = tf.nn.relu(normalized2, name='relu_2')
    return result_relu_2

def dk(input, k, is_train,reuse=False, norm='instance', is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",is_train = is_train,
      shape=[3, 3, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training = is_training,norm= norm,is_train = is_train)
    output = tf.nn.relu(normalized)
    return output

def Rk(input, k,  is_train, reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",is_train = is_train,
        shape=[3, 3, input.get_shape()[3], k])
      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training = is_training,norm= norm, is_train = is_train)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",is_train = is_train,
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_train = is_train,is_training = is_training,norm= norm)
    output = input+normalized2
    return output

def n_res_blocks(input, is_train,reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
#    if i==4 or i==6 or i==9 :
#      output = ladain(input, mean1=mean1,var1=var1,mean2=mean2,var2=var2)
#    else:
    output = Rk(input, depth, is_train = is_train,reuse = reuse, is_training = is_training,norm= norm, name='R{}_{}'.format(depth, i))
    input = output
  return output

def uk(input, k, is_train,reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
      with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",is_train=is_train,
      shape=[3, 3, k, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_train=is_train,is_training = is_training,norm= norm)
    output = tf.nn.relu(normalized)
    return output

### Discriminator layers
def Ck(input, k,is_train, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    slope: LeakyReLU's slope
    stride: integer
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'C64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",is_train=is_train,
      shape=[4, 4, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_train=is_train,is_training = is_training,norm= norm)
    output = _leaky_relu(normalized, slope)
    return output

def last_conv(input,is_train, reuse=False, use_sigmoid=False, name=None):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",is_train=is_train,
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases",is_train=is_train, shape = [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output

### Helpers
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

def _biases(name, is_train,shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,trainable=is_train,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_train,is_training, norm='instance',name = 'norm'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input,is_train = is_train,name=name)
  elif norm == 'batch':
    return _batch_norm(input, is_train = is_train,is_training = is_training,name=name)
  else:
    return input

def _batch_norm(input,is_train, is_training,name = 'norm'):
  """ Batch Normalization
  """
  with tf.variable_scope(name+"_batch_norm"):
    return tf.layers.batch_normalization(input,
             #                           decay=0.9,
                                        scale=True,
             #                           updates_collections=None,
                                        trainable = is_train,
                                        training=is_training)

def _instance_norm(input,is_train,name = 'norm'):
  """ Instance Normalization
  """
  with tf.variable_scope(name+"_instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", is_train = is_train,shape =[depth], mean=1.0)
    offset = _biases("offset",is_train = is_train,shape = [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
