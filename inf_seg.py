import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from metric import dice_score2
from metric import sensitivity
from metric import haff 
import aug
import numpy as np
import scipy.io
import SimpleITK as sitk
from model import GAN
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model5', 'pretrained/proposed.pb', 'model path (.pb)')

tf.flags.DEFINE_integer('image_size', '400', 'image size, default: 400')
tf.flags.DEFINE_string('valPath', './test/',
                       'validata path, default:')
tf.flags.DEFINE_string('rsltPath', './result_label/',
                       'result path, default:')
def inference():
  graph = tf.Graph()
  list2 = os.listdir(FLAGS.valPath) 
  with graph.as_default():
    y = tf.placeholder(tf.float32,
        shape=[1, FLAGS.image_size, FLAGS.image_size, 1])
    z = tf.placeholder(tf.float32,
        shape=[1, FLAGS.image_size, FLAGS.image_size, 1])

    with tf.gfile.FastGFile(FLAGS.model5, 'rb') as model_file5:
        graph_def5 = tf.GraphDef()
        graph_def5.ParseFromString(model_file5.read())
        
#    output_image = tf.import_graph_def(graph_def,
#                    input_map={'y': y,'z': z,'segy': segy,'segz': segz},
#                    return_elements=['output:0'],
#                    name="output")[0]

    segRSLT5 = tf.import_graph_def(graph_def5,input_map={'y': y},return_elements=['layer_9:0'],name="layer_9")[0]
                    
#    output_image = tf.cast(output_image,tf.float32)
  sess = tf.Session(graph=graph)
  count = 0
  for i in range(len(list2)):
#    file_names = list[i].split("_")
#    if file_names[0] == '3' and file_names[2] == 'i1' and file_names[4] == '1.mat':
#    if list[i] == '10_19.mat':
      print('!!!!!')
      path = os.path.join(FLAGS.valPath,list2[i])
      dt = scipy.io.loadmat(path)
      y_val = np.float32(dt['y'])
      y_val = y_val.astype(np.float32)

      z_val = np.float32(dt['z'])
      z_val = np.squeeze(z_val)
      
      x_val = np.float32(dt['x'])
      x_val = np.squeeze(x_val)
      x_val = aug.L2HOT(x_val,8)
      
      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis]

      generated5 = segRSLT5.eval(session = sess,feed_dict={y:y_val})
      
      x_val = np.squeeze(x_val)
      inds = np.argmax(x_val,axis = 2)
      inds = np.squeeze(inds) 
      
      y_val = np.squeeze(y_val)

      generated5 = np.squeeze(generated5)
      

      inds5 = np.argmax(generated5,axis = 2)
      inds5 = np.squeeze(inds5)
      x_val = (x_val/10)*255
      inds5 = (inds5/10)*255
      inds = (inds/10)*255
      y_val = (y_val)*255
      img = np.hstack([y_val,inds5,inds])  
#      cv2.imwrite(os.path.join(FLAGS.rsltPath,path.split('/')[-1].split('.')[0]+'.png'), img)    
      plt.imsave(os.path.join(FLAGS.rsltPath,path.split('/')[-1].split('.')[0]+'.png'),img,format='png',cmap='plasma')
      
def main(unused_argv):
  inference()

if __name__ == '__main__':

  tf.app.run()      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
  
  
  
  
  
  