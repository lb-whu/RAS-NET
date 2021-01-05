import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ops
from discriminator import Discriminator
from generator import Generator
from Unet import Unet

REAL_LABEL = 0.9

class GAN:
  def __init__(self,
               is_trainGAN = True,
               is_trainUnet = True,
               batch_size=1,
               image_size=400,
               use_lsgan=True,
               norm='instance',
               learning_rate=2e-4,
               beta1=0.5,
               ngf=64,
               num_class = 8,
              ):
    """
    Args:
      batch_size: integer, batch size
      image_size: integer, image size
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.zf = 0.1
    self.yf = 1
    self.shapef = 0.5

    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.is_trainGAN = is_trainGAN
    self.is_trainUnet = is_trainUnet
    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')


    self.Unet = Unet('Unet', is_train = self.is_trainUnet ,is_training = self.is_training, ngf=ngf, norm=norm, num_class=num_class,image_size=image_size)
    self.G = Generator('G', is_train = self.is_trainGAN ,is_training = self.is_training, ngf=ngf, norm=norm, image_size=image_size)

    self.D_Z = Discriminator('D_Z',is_train = self.is_trainGAN ,
            is_training = self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.D_Z1 = Discriminator('D_Z1',is_train = self.is_trainGAN ,
            is_training = self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.D_Z2 = Discriminator('D_Z2',is_train = self.is_trainGAN ,
            is_training = self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 1])
    self.fake_y1 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/2), int(image_size/2), 1])
    self.fake_y2 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 1])
        
            
    self.x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 8])
    self.zx = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 8])
        
    self.y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 1])
        
    self.mean1 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 256])
    self.var1 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 256])
    self.mean2 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 256])
    self.var2 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 256])
        
        
    self.z = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 1])
    self.z1 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/2), int(image_size/2), 1])
    self.z2 = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 1])
    self.segy = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 8])
    self.segz = tf.placeholder(tf.float32,
        shape=[batch_size, int(image_size/4), int(image_size/4), 8])
#    self.fake_z = tf.placeholder(tf.float32,
#        shape=[batch_size, image_size, image_size, 3])

  def model(self):  


    fake_y,fake_y1,fake_y2,feat = self.G(self.y,self.mean1,self.var1,self.mean2,self.var2) 

#    sobel_fakey = self.sober(fake_y)
#    sobel_y = self.sober(y)                                                    
#    D_Y_loss = self.discriminator_loss(self.D_Y, sobel_y, sobel_fakey, use_lsgan=self.use_lsgan)*self.zf
    
    seg = self.Unet(self.y)
    G_gan_loss_z = self.generator_loss(self.D_Z, fake_y, use_lsgan=self.use_lsgan)*self.zf
    G_gan_loss_z1 = self.generator_loss(self.D_Z1, fake_y1, use_lsgan=self.use_lsgan)*self.zf
    G_gan_loss_z2 = self.generator_loss(self.D_Z2, fake_y2, use_lsgan=self.use_lsgan)*self.zf
    G_gan_loss = 0.15*G_gan_loss_z+0.3*G_gan_loss_z1+0.45*G_gan_loss_z2
    
    D_Z_loss = self.discriminator_loss(self.D_Z, self.z, self.fake_y, use_lsgan=self.use_lsgan)*self.zf
    D_Z_loss1 = self.discriminator_loss(self.D_Z1, self.z1, self.fake_y1, use_lsgan=self.use_lsgan)*self.zf
    D_Z_loss2 = self.discriminator_loss(self.D_Z2, self.z2, self.fake_y2, use_lsgan=self.use_lsgan)*self.zf
    D_loss = 0.15*D_Z_loss+0.3*D_Z_loss1+0.45*D_Z_loss2
    
    DiceTrain = self.unetLoss(self.Unet,self.x,self.y)
    DiceLoss = self.unetLoss(self.Unet,self.x,fake_y)*0.6
        
        
    return G_gan_loss, D_loss,fake_y,DiceLoss,DiceTrain,seg,feat

  def optimize(self, G_loss, D_Z_loss,DiceLoss):#H_loss,D_Z_loss
    def make_optimizer(loss, variables, name='Adam'):
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
#      print('variables',variables)
      gradients = optimizer.compute_gradients(loss,var_list=variables,colocate_gradients_with_ops=True)

      learning_step = (
              optimizer.apply_gradients(gradients)
              )

      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Z_optimizer = make_optimizer(D_Z_loss, self.D_Z.variables, name='Adam_D_Z')
    D_Z_optimizer1 = make_optimizer(D_Z_loss, self.D_Z1.variables, name='Adam_D_Z1')
    D_Z_optimizer2 = make_optimizer(D_Z_loss, self.D_Z2.variables, name='Adam_D_Z2')
    Dice_optimizer = make_optimizer(DiceLoss, self.G.variables, name='Adam_dice')

    with tf.control_dependencies([G_optimizer, Dice_optimizer, D_Z_optimizer, D_Z_optimizer1, D_Z_optimizer2]):#H_optimizer,D_Z_optimizer
      return tf.no_op(name='optimizers')
      
  def optimize3(self, G_loss,DiceLoss):#H_loss,D_Z_loss
    def make_optimizer(loss, variables, name='Adam3'):
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
#      print('variables',variables)
      gradients = optimizer.compute_gradients(loss,var_list=variables,colocate_gradients_with_ops=True)

      learning_step = (
              optimizer.apply_gradients(gradients)
              )

      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G3')
    Dice_optimizer = make_optimizer(DiceLoss, self.G.variables, name='Adam_dice3')

    with tf.control_dependencies([G_optimizer, Dice_optimizer]):#H_optimizer,D_Z_optimizer
      return tf.no_op(name='optimizers3')
      
  def optimize2(self, DiceTrain):#H_loss,D_Z_loss
    def make_optimizer(loss, variables, name='Adam2'):
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
#      print('variables',variables)
      gradients = optimizer.compute_gradients(loss,var_list=variables,colocate_gradients_with_ops=True)

      learning_step = (
              optimizer.apply_gradients(gradients)
              )

      #learning_step = (
      #    tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
      #            .minimize(loss, global_step=global_step, var_list=variables)
      #)
      return learning_step

    DiceTrain_optimizer = make_optimizer(DiceTrain, self.Unet.variables, name='Adam_unet')
    with tf.control_dependencies([DiceTrain_optimizer]):#H_optimizer,D_Z_optimizer
      return tf.no_op(name='optimizers2')
      

  def sober(self,source):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    filtered_x = tf.nn.conv2d(source, sobel_x_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(source, sobel_y_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    s = tf.sqrt(filtered_x*filtered_x + filtered_y*filtered_y)
    return s
  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      print(fake_y)
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
      print(fake_y)
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss
    
  def discriminator_loss_x_z(self, D, y, fake_y,x, use_lsgan=True):
    fake_x_y = tf.concat( [fake_y, x],3)
    x_y = tf.concat( [y, x],3)
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(x_y), REAL_LABEL))
      print(fake_y)
      error_fake = tf.reduce_mean(tf.square(D(fake_x_y)))
      print(fake_y)
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss
  def generator_loss(self, D, fake_y, use_lsgan=True):
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss
  def generator_loss_x_z(self, D, fake_y, x,use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    fake_x_y = tf.concat( [fake_y, x],3)
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_x_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_x_y))) / 2
    return loss
    

  def unetLoss(self,Unet, x,y):
    e=1e-5
    layer_seg1 = Unet(y)
    loss_type = 'jaccard'
    inse1 = tf.reduce_sum(layer_seg1 * x, axis=(1,2,3))
    if loss_type == 'jaccard':
        l1 = tf.reduce_sum(layer_seg1 * layer_seg1, axis=(1,2,3))
        r1 = tf.reduce_sum(x * x, axis=(1,2,3))
    else:
        l1 = tf.reduce_sum(layer_seg1, axis=(1,2,3))
        r1 = tf.reduce_sum(x, axis=(1,2,3))

    dice1 = (2. * inse1 + e) / (l1 + r1 + e)
    dice1 = tf.reduce_mean(dice1)
    
    return 1 -dice1 
  def diceSIM(self, x,y,Lx,Ly):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    e=1e-5

    
    xx = tf.concat([x,x,x,x,x,x,x,x],3)
    yy = tf.concat([y,y,y,y,y,y,y,y],3)
        
    inse1 = tf.reduce_sum(Lx * xx, axis=(1,2))
    inse2 = tf.reduce_sum(Ly * yy, axis=(1,2))

    l1 = tf.reduce_sum(Lx, axis=(1,2))
    l2 = tf.reduce_sum(Ly, axis=(1,2))
    
    diceDIS = tf.reduce_mean(tf.abs((inse1/(l1+e)) - (inse2/(l2+e))))
    
    return diceDIS
    
    
    
    
    
    
    
    
    
    
    
