import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import nets

class Unet:
  def __init__(self, name, is_train,is_training, ngf=64, norm='instance', num_class = 8,image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.num_class = num_class
    self.is_train = is_train
    self.result_from_contract_layer = {}
  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c1,c2,c3,c4,layer_5 = self.downsample(input)
      
      # mid layer
      layer_mid = nets.layer_mid(layer_5,16*self.ngf,8*self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                                keep_prob= 1,name = 'layer_mid') 

      # deconv layers
      layer_6 = nets.layer_up(c4,layer_mid,8*self.ngf,4*self.ngf,is_train=self.is_train,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = 'layer_6')
      
      layer_7 = nets.layer_up(c3,layer_6,4*self.ngf,2*self.ngf,is_train=self.is_train,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = 'layer_7')
                                
      layer_8 = nets.layer_up(c2,layer_7,2*self.ngf,self.ngf,is_train=self.is_train,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = 'layer_8')
                                
      layer_9 = nets.layer_last(c1,layer_8,self.ngf,self.num_class,is_train=self.is_train,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = 'layer_9')
      
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return layer_9
  def downsample(self, input):
    with tf.variable_scope(self.name):
      c1,layer_1 = nets.layer_down(input,self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                            keep_prob= 1,name = 'layer_1')

      c2,layer_2 = nets.layer_down(layer_1,2*self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                            keep_prob= 1,name = 'layer_2')

      c3,layer_3 = nets.layer_down(layer_2,4*self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                            keep_prob= 1,name = 'layer_3')     

      c4,layer_4 = nets.layer_down(layer_3,8*self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                            keep_prob= 1,name = 'layer_4')
      
      layer_5,_ = nets.layer_down(layer_4,16*self.ngf,is_train=self.is_train,is_training=self.is_training,reuse=self.reuse,
                            keep_prob= 1,name = 'layer_5')
      return c1,c2,c3,c4,layer_5

