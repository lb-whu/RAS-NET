""" 
Freeze variables and convert 2 generator networks 
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import GAN
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 400, 'image size, default: 400')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'batch',
                       '[instance, batch] use instance norm  or batch norm, default: instance')

tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
                      
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('class_num', 8,
                        'number of gen filters in first conv layer, default: 64')
                        
tf.flags.DEFINE_string('model1', 'gen_proposed.pb', 'XtoY model name, default: s2t.pb')
tf.flags.DEFINE_string('model2', 'feat.pb', 'XtoY model name, default: s2t.pb')
tf.flags.DEFINE_string('checkpoint_dir',None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
                        

def export_graph(model_name1,model_name2):
  graph = tf.Graph()

  with graph.as_default():
    ad_gan = GAN(
        is_trainGAN = True,
        is_trainUnet = True,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        num_class = FLAGS.class_num
    )
    y = tf.placeholder(tf.float32,
        shape=[1, FLAGS.image_size, FLAGS.image_size, 1],name='y')
    mean1 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256],name='m1')
    var1 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256],name='v1')
    mean2 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256],name='m2')
    var2 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256],name='v2')
    ad_gan.model()
    
    output_image,feat = ad_gan.G.sample(y,mean1,var1,mean2,var2)
    output_image = tf.abs(output_image)
    
#    layer_9 = ad_gan.Unet(y)
#    if XtoY:
#      output_image = cycle_gan.G.sample(tf.expand_dims(input_image1, 0),tf.expand_dims(input_image2, 0))
#    else:
#      output_image = cycle_gan.F.sample(tf.expand_dims(input_image1, 0))

    output_image = tf.identity(output_image, name='output')
    feat = tf.identity(feat, name='R256_8')
#    layer_9 = tf.identity(layer_9, name='layer_9')
    
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])
    output_graph_feat = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [feat.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name1, as_text=False)
    tf.train.write_graph(output_graph_feat, 'pretrained', model_name2, as_text=False)

def main(unused_argv):
  print('Export model...')
  export_graph(FLAGS.model1,FLAGS.model2)


if __name__ == '__main__':
  tf.app.run()
