import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

import aug
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from model import GAN
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/gen_proposed.pb', 'model path (.pb)')
tf.flags.DEFINE_string('model1', 'pretrained/feat.pb', 'model path (.pb)')
tf.flags.DEFINE_string('model2', 'pretrained/proposed.pb', 'model path (.pb)')

tf.flags.DEFINE_integer('image_size', '400', 'image size, default: 400')
tf.flags.DEFINE_string('valPath', './val/',
                       'validata path, default:')

tf.flags.DEFINE_string('rsltPath', './val2/',
                       'validata path, default:')
def inference():
  graph = tf.Graph()
  list = os.listdir(FLAGS.valPath)  
  with graph.as_default():
    y = tf.placeholder(tf.float32,
        shape=[1, FLAGS.image_size, FLAGS.image_size, 1])
    mean1 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256])
    var1 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256])
    mean2 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256])
    var2 = tf.placeholder(tf.float32,
        shape=[1, int(FLAGS.image_size/4), int(FLAGS.image_size/4), 256])
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
    with tf.gfile.FastGFile(FLAGS.model1, 'rb') as model_file1:
        graph_def1 = tf.GraphDef()
        graph_def1.ParseFromString(model_file1.read())    
    with tf.gfile.FastGFile(FLAGS.model2, 'rb') as model_file2:
        graph_def2 = tf.GraphDef()
        graph_def2.ParseFromString(model_file2.read())

            
    output_image = tf.import_graph_def(graph_def,
                    input_map={'y': y,'m1': mean1,'m2': mean2},
                    return_elements=['output:0'],
                    name="output")[0]
                    
                    
    feat = tf.import_graph_def(graph_def1,
                    input_map={'y': y},
                    return_elements=['R256_8:0'],
                    name="R256_8")[0]
                    
    segRSLT = tf.import_graph_def(graph_def2,input_map={'y': y},return_elements=['layer_9:0'],name="layer_9")[0]
    output_image = tf.cast(output_image,tf.float32)
  sess = tf.Session(graph=graph)
  count = 0
  for i in range(len(list)):
#    file_names = list[i].split("_")
#    if file_names[0] == '3' and file_names[2] == 'i1' and file_names[4] == '1.mat':
#    if list[i] == '10_19.mat':
      path = os.path.join(FLAGS.valPath,list[i])
      dt = scipy.io.loadmat(path)
      y_val = np.float32(dt['y'])
      y_val = y_val.astype(np.float32)

      z_val = np.float32(dt['z'])
      z_val = np.squeeze(z_val)
      
      x_val = np.float32(dt['x'])
      x_val = np.squeeze(x_val)
      x_val = aug.L2HOT(x_val,8)
      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis]
      
      zx_val = sess.run(segRSLT,feed_dict={y: z_val})
      zx_val = np.squeeze(zx_val)
      inds = np.argmax(zx_val,axis = 2)
      zx_val = aug.L2HOT(inds,8)
      zx_val = zx_val[np.newaxis,:,:,:]
      zx_val = aug.align_label(x_val,zx_val) 
      
      y_val = np.squeeze(y_val);z_val = np.squeeze(z_val);x_val = np.squeeze(x_val);zx_val = np.squeeze(zx_val)
      meany_img,vary_img,meanz_img,varz_img = aug.getMeanVarIMG(y_val,z_val,x_val,zx_val,8)
                
      ft_y_m = sess.run(feat,feed_dict={y: meany_img})
#      ft_y_v = sess.run(feat,feed_dict={ad_gan.y: vary_img})
      ft_z_m = sess.run(feat,feed_dict={y: meanz_img})
#      ft_z_v = sess.run(feat,feed_dict={ad_gan.y: varz_img})
      
      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis];zx_val = zx_val[np.newaxis,:,:,:]
      generated = output_image.eval(session = sess,feed_dict={y: y_val,mean1: ft_y_m,mean2: ft_z_m})
      x_val = np.squeeze(x_val)
      y_val = np.squeeze(y_val)
      z_val = np.squeeze(z_val)
      generated = np.squeeze(generated)
      generated[generated>0.85] = 0.85
      scipy.io.savemat(os.path.join(FLAGS.rsltPath,path.split('/')[-1]),{'img':generated,'x':x_val,'y':y_val,'z':z_val}) 
      
#      img = np.hstack([y_val,z_val,generated,generatedcy,img_val])      
      plt.imsave(os.path.join(FLAGS.rsltPath,path.split('/')[-1].split('.')[0]+'_y.png'),y_val,format='png',cmap='plasma')
      plt.imsave(os.path.join(FLAGS.rsltPath,path.split('/')[-1].split('.')[0]+'_z.png'),z_val,format='png',cmap='plasma')
      plt.imsave(os.path.join(FLAGS.rsltPath,path.split('/')[-1].split('.')[0]+'_prop.png'),generated,format='png',cmap='plasma')
      
      count = count + 1
  count = 0



def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
