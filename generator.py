import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ops

class Generator:
  def __init__(self, name, is_train,is_training, ngf=64, norm='instance', image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.is_train = is_train

  def __call__(self, input1,mean1,var1,mean2,var2):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32_1 = ops.c7s1_k(input1, self.ngf,is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32_1')                             # (?, w, h, 64)
      d64_1 = ops.dk(c7s1_32_1, 2*self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64_1')                                 # (?, w/2, h/2, 128)
      d128_1 = ops.dk(d64_1, 4*self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128_1')                                # (?, w/4, h/4, 256)
          
      
          
#      c7s1_32_3 = ops.c7s1_k(input2, self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
#          reuse=self.reuse, name='c7s1_32_3')                             # (?, w, h, 64)
#      d64_3 = ops.dk(c7s1_32_2, 2*self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
#          reuse=self.reuse, name='d64_3')                                 # (?, w/2, h/2, 128)
#      d128_3 = ops.dk(d64_2, 4*self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
#          reuse=self.reuse, name='d128_3')                                # (?, w/4, h/4, 256)  
          
#      d128 = tf.concat( [d128_1, d128_2],3)    
      res_output = ops.n_res_blocks(d128_1,is_train=self.is_train ,reuse=self.reuse, n=8)
      ladin_output = ops.ladain(res_output, mean1=mean1,var1=var1,mean2=mean2,var2=var2)
      
#      res_output = ops.n_res_blocks(d128_1,mean1 = d128_mean1,var1 = d128_var1,mean2 = d128_mean2, var2 = d128_var2,is_train=self.is_train ,reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)
      
#      d128 = tf.concat( [res_output, d128_3],3)  
#      ad128 = ops.attention(d128, 8*self.ngf, is_training=self.is_training,
#          reuse=self.reuse, name='ad128')
            
      # fractional-strided convolution

      u64 = ops.uk(ladin_output, 2*self.ngf,is_train=self.is_train , is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 128)
      u32 = ops.uk(u64, self.ngf, is_train=self.is_train ,is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 64)


      output = ops.c7s1_k(u32, 1, is_train=self.is_train ,norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 1)
      output1 = tf.nn.avg_pool(value=output, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='VALID', name='avgpool1')       # (?, w/2, h/2, 1)
      output2 = tf.nn.avg_pool(value=output1, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='VALID', name='avgpool2')       # (?, w/4, h/4, 1)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return output,output1,output2,res_output
  def sample(self, input1,mean1,var1,mean2,var2):
    image,_,_,res_output = self.__call__(input1,mean1,var1,mean2,var2)
    return image,res_output
