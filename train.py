import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import GAN
from datetime import datetime
import os
import logging
import random
import numpy as np
import aug

import matplotlib.pyplot as plt
import scipy.io as sio 
try:
  from os import scandir
except ImportError:
  from scandir import scandir
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
                        
tf.flags.DEFINE_string('DataPath', './train/',
                       'data set path, default:')
tf.flags.DEFINE_string('valPath', './val/',
                       'validata path, default:')

tf.flags.DEFINE_string('load_model',None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
                        
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
#    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
def train():
  TTFLAG = 1
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)

    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  
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
    G_gan_loss_z, D_Z_loss,fake_y,DiceLoss,DiceTrain,seg,feat = ad_gan.model()
    optimizersU = ad_gan.optimize2(DiceTrain)
    optimizers = ad_gan.optimize(G_gan_loss_z,D_Z_loss,DiceLoss)
#    optimizers3 = ad_gan.optimize3(G_gan_loss_z,DiceLoss)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()
  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      print(meta_graph_path)
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      num_samples = 8000
      loss_1000_G = 0
      loss_1000_ALL = 0
      loss_1000_D = 0
      loss_1000_Dice = 0
      loss_echo = 0
      flag = 1
      G_z_val = 0 
      D_z_val = 0
      Dice_val = 0
      count = 0
      swit = 1
      val_files = data_reader(FLAGS.valPath)
      while not coord.should_stop():
        file_paths_X = data_reader(FLAGS.DataPath)
        for i in range(len(file_paths_X)):
          if not coord.should_stop():
            file_path_X = file_paths_X[i]
            data=sio.loadmat(file_path_X) 
            y_val = np.float32(data['y'])
            y_val = y_val.astype(np.float32)
      
            x_val = np.float32(data['x'])
            x_val = np.squeeze(x_val)
            
            z_val = np.float32(data['z'])
            z_val = np.squeeze(z_val)
            
            x_val,y_val,z_val = aug.aug2D(x_val,y_val,z_val,FLAGS.class_num)
            
            
            x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis]
            
            z_val1 = aug.zm(z_val)
            z_val2 = aug.zm(z_val1)
            

            if step < 80000:#train unet
   
#              fakey_val = sess.run(fake_y,feed_dict={cycle_gan.y: y_val,cycle_gan.z:z_val})
              _, Dice_val,summary = (
                  sess.run([optimizersU, DiceTrain,summary_op],feed_dict={ad_gan.x: x_val,ad_gan.y: y_val}))

            elif (step >= 80000):#train GAN
              if flag == 1:
                if swit == 1:
                  swit = 0
#                  intG = tf.variables_initializer(ad_gan.G.variables)
#                  intDZ = tf.variables_initializer(ad_gan.D_Z.variables)
#                  intDZ1 = tf.variables_initializer(ad_gan.D_Z1.variables)
#                  intDZ2 = tf.variables_initializer(ad_gan.D_Z2.variables)
#                  sess.run([intG, intDZ,intDZ1,intDZ2])
                
                zx_val = sess.run(seg,feed_dict={ad_gan.y: z_val})
                zx_val = np.squeeze(zx_val)
                inds = np.argmax(zx_val,axis = 2)
                zx_val = aug.L2HOT(inds,FLAGS.class_num)
                zx_val = zx_val[np.newaxis,:,:,:]
                zx_val = aug.align_label(x_val,zx_val) 
                
                y_val = np.squeeze(y_val);z_val = np.squeeze(z_val);x_val = np.squeeze(x_val);zx_val = np.squeeze(zx_val)
                meany_img,vary_img,meanz_img,varz_img = aug.getMeanVarIMG(y_val,z_val,x_val,zx_val,8)
                ft_y_m = sess.run(feat,feed_dict={ad_gan.y: meany_img})
                ft_y_v = sess.run(feat,feed_dict={ad_gan.y: vary_img})
                ft_z_m = sess.run(feat,feed_dict={ad_gan.y: meanz_img})
                ft_z_v = sess.run(feat,feed_dict={ad_gan.y: varz_img})
                x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis];zx_val = zx_val[np.newaxis,:,:,:]
                
                
                
                fakey_val = sess.run(fake_y,feed_dict={ad_gan.y: y_val,ad_gan.z:z_val,ad_gan.mean1: ft_y_m,ad_gan.var1: ft_y_v,ad_gan.mean2: ft_z_m,ad_gan.var2: ft_z_v})
                fakey_val1 = aug.zm(fakey_val)
                fakey_val2 = aug.zm(fakey_val1)
                
                
                _, G_z_val, D_z_val,Dice_val,summary = (
                    sess.run([optimizers, G_gan_loss_z, D_Z_loss,DiceLoss,summary_op],
                          feed_dict={ad_gan.x: x_val,ad_gan.y: y_val,ad_gan.z:z_val,ad_gan.mean1: ft_y_m,ad_gan.var1: ft_y_v,ad_gan.mean2: ft_z_m,ad_gan.var2: ft_z_v,
                                      ad_gan.z1:z_val1,ad_gan.z2:z_val2,ad_gan.fake_y:fakey_val,ad_gan.fake_y1:fakey_val1,ad_gan.fake_y2:fakey_val2,ad_gan.zx:zx_val}))
                                      
#                zx_val = np.squeeze(zx_val)
#                inds = np.argmax(zx_val,axis = 2)                      
#                meany_img = np.squeeze(meany_img);vary_img = np.squeeze(vary_img);meanz_img = np.squeeze(meanz_img);varz_img = np.squeeze(varz_img);zx_val = np.squeeze(zx_val)
#                sio.savemat(os.path.join('./result/',str(step)+'_'+file_path_X.split('/')[-1]),{'m1':meany_img,'v1':vary_img,'m2':inds,'v2':varz_img},do_compression = True)
                if step % 50 == 0:
                  csum = 0
                  cmean = 0
                  cvar = 0
                  for n in range(len(val_files)):
                      val_file = val_files[n]
                      dt = sio.loadmat(val_file) 
                      
                      y_val = np.float32(dt['y'])
                      y_val = y_val.astype(np.float32)

                      z_val = np.float32(dt['z'])
                      z_val = np.squeeze(z_val)
                      z_val3 = z_val.copy()
                      
                      x_val = np.float32(dt['x'])
                      x_val = np.squeeze(x_val)
                      x_val3 = x_val.copy()
                      x_val = aug.L2HOT(x_val,FLAGS.class_num)
                      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis]                      

                      
                      zx_val = sess.run(seg,feed_dict={ad_gan.y: z_val})
                      zx_val = np.squeeze(zx_val)
                      inds = np.argmax(zx_val,axis = 2)
                      zx_val3 = inds.copy()
                      zx_val = aug.L2HOT(inds,8)
                      zx_val = zx_val[np.newaxis,:,:,:]

                      zx_val = aug.align_label(x_val,zx_val) 
                      y_val = np.squeeze(y_val);z_val = np.squeeze(z_val);x_val = np.squeeze(x_val);zx_val = np.squeeze(zx_val)
                      meany_img,vary_img,meanz_img,varz_img = aug.getMeanVarIMG(y_val,z_val,x_val,zx_val,8)
                      ft_y_m = sess.run(feat,feed_dict={ad_gan.y: meany_img})
                      ft_y_v = sess.run(feat,feed_dict={ad_gan.y: vary_img})
                      ft_z_m = sess.run(feat,feed_dict={ad_gan.y: meanz_img})
                      ft_z_v = sess.run(feat,feed_dict={ad_gan.y: varz_img})
                      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis];zx_val = zx_val[np.newaxis,:,:,:]
                      
                      fakey_val = sess.run(fake_y,feed_dict={ad_gan.y: y_val,ad_gan.z:z_val,ad_gan.mean1: ft_y_m,ad_gan.var1: ft_y_v,ad_gan.mean2: ft_z_m,ad_gan.var2: ft_z_v})
                      
#                      is_finished = aug.isfinish()
                      
                      fakey_val2 = fakey_val.copy()
                      fakey_val[(y_val>0.05)|(z_val>0.05)] = 0
                      fakey_val = np.squeeze(fakey_val)
                      x_val = np.squeeze(x_val)
                      fakey_val2 = np.squeeze(fakey_val2)
                      mmean,mvar = aug.getMeanVar(fakey_val2,z_val3,x_val3,zx_val3,FLAGS.class_num)
                      sum1 = np.sum(np.abs(fakey_val.flatten()))
                      csum = csum + sum1
                      cvar = cvar + mvar
                      cmean = cmean + mmean
                      fakey_val2 = np.squeeze(fakey_val2)
                      x_val = np.squeeze(x_val)
                      y_val = np.squeeze(y_val)
                      z_val = np.squeeze(z_val)
                      fakey_val2 = np.abs(fakey_val2)
                      if val_file.split('/')[-1].split('.')[0] == '12_3_30':
                        plt.imsave(os.path.join('./result/',str(step)+'_'+val_file.split('/')[-1].split('.')[0]+'f.png'),fakey_val2,format='png',cmap='plasma')
                        plt.imsave(os.path.join('./result/',val_file.split('/')[-1].split('.')[0]+'y.png'),y_val,format='png',cmap='plasma')
                        plt.imsave(os.path.join('./result/',val_file.split('/')[-1].split('.')[0]+'z.png'),z_val,format='png',cmap='plasma')
#                        cv2.imwrite(os.path.join('./result/',str(step)+'_'+val_file.split('/')[-1].split('.')[0]+'f.png'), fakey_val2*255)
#                        cv2.imwrite(os.path.join('./result/',val_file.split('/')[-1].split('.')[0]+'y.png'), y_val*255)
#                        cv2.imwrite(os.path.join('./result/',val_file.split('/')[-1].split('.')[0]+'z.png'), z_val*255)
#                        sio.savemat(os.path.join('./result/',str(step)+'_'+val_file.split('/')[-1]),{'img':fakey_val2,'x':x_val,'y':y_val,'z':z_val},do_compression = True) 
                  logger.info('  csum   : {}'.format(csum))
                  logger.info('  cmean   : {}'.format(cmean))   
                  logger.info('  cvar  : {}'.format(cvar))
                  
                  is_finished = aug.isfinish(csum,cmean,cvar)
                  if (step>82000) & (cvar < 1.5):
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                  if is_finished:
                    for n in range(len(val_files)):
                      val_file = val_files[n]
                      dt = sio.loadmat(val_file) 
                      
                      y_val = np.float32(dt['y'])
                      y_val = y_val.astype(np.float32)

                      z_val = np.float32(dt['z'])
                      z_val = np.squeeze(z_val)
                      z_val3 = z_val.copy()
                      
                      x_val = np.float32(dt['x'])
                      x_val = np.squeeze(x_val)
                      x_val3 = x_val.copy()
                      x_val = aug.L2HOT(x_val,FLAGS.class_num)
                      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis]                      
                      
                      zx_val = sess.run(seg,feed_dict={ad_gan.y: z_val})
                      zx_val = np.squeeze(zx_val)
                      inds = np.argmax(zx_val,axis = 2)
                      zx_val3 = inds.copy()
                      zx_val = aug.L2HOT(inds,8)
                      zx_val = zx_val[np.newaxis,:,:,:]
                      
                      zx_val = aug.align_label(x_val,zx_val)                 
                      y_val = np.squeeze(y_val);z_val = np.squeeze(z_val);x_val = np.squeeze(x_val);zx_val = np.squeeze(zx_val)
                      meany_img,vary_img,meanz_img,varz_img = aug.getMeanVarIMG(y_val,z_val,x_val,zx_val,8)
                      ft_y_m = sess.run(feat,feed_dict={ad_gan.y: meany_img})
                      ft_y_v = sess.run(feat,feed_dict={ad_gan.y: vary_img})
                      ft_z_m = sess.run(feat,feed_dict={ad_gan.y: meanz_img})
                      ft_z_v = sess.run(feat,feed_dict={ad_gan.y: varz_img})
                      x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis];zx_val = zx_val[np.newaxis,:,:,:]
                      
                      fakey_val = sess.run(fake_y,feed_dict={ad_gan.y: y_val,ad_gan.z:z_val,ad_gan.mean1: ft_y_m,ad_gan.var1: ft_y_v,ad_gan.mean2: ft_z_m,ad_gan.var2: ft_z_v})
                      fakey_val = np.squeeze(fakey_val)
                      x_val = np.squeeze(x_val)
                      y_val = np.squeeze(y_val)
                      z_val = np.squeeze(z_val)
                      sio.savemat(os.path.join('./result/','f' + val_file.split('/')[-1]),{'img':fakey_val,'x':x_val,'y':y_val,'z':z_val},do_compression = True) 
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    flag = 0
                    swit = 1
              else:
                if swit == 1:
                  swit = 0 
                  if TTFLAG == 0:
                    count = 5995
                    TTFLAG = 1
                  else:
                    count = 0
                  print('run seg')
                  print(step)
                zx_val = sess.run(seg,feed_dict={ad_gan.y: z_val})
                zx_val = np.squeeze(zx_val)
                inds = np.argmax(zx_val,axis = 2)
                zx_val = aug.L2HOT(inds,8)
                zx_val = zx_val[np.newaxis,:,:,:]
                
                zx_val = aug.align_label(x_val,zx_val)                 
                y_val = np.squeeze(y_val);z_val = np.squeeze(z_val);x_val = np.squeeze(x_val);zx_val = np.squeeze(zx_val)
                meany_img,vary_img,meanz_img,varz_img = aug.getMeanVarIMG(y_val,z_val,x_val,zx_val,8)
                ft_y_m = sess.run(feat,feed_dict={ad_gan.y: meany_img})
                ft_y_v = sess.run(feat,feed_dict={ad_gan.y: vary_img})
                ft_z_m = sess.run(feat,feed_dict={ad_gan.y: meanz_img})
                ft_z_v = sess.run(feat,feed_dict={ad_gan.y: varz_img})
                x_val = x_val[np.newaxis,:,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis];zx_val = zx_val[np.newaxis,:,:,:]
                
                if step % 2 == 0:   #50% original samples
                    _, Dice_val,summary = (
                        sess.run([optimizersU, DiceTrain,summary_op],feed_dict={ad_gan.x: x_val,ad_gan.y: y_val}))
                else:
                    fakey_val = sess.run(fake_y,feed_dict={ad_gan.y: y_val,ad_gan.z:z_val,ad_gan.mean1: ft_y_m,ad_gan.var1: ft_y_v,ad_gan.mean2: ft_z_m,ad_gan.var2: ft_z_v})
                    fakey_val = np.abs(fakey_val)
                    _, Dice_val,summary = (
                        sess.run([optimizersU, DiceTrain,summary_op],feed_dict={ad_gan.x: x_val,ad_gan.y: fakey_val}))
                count += 1
                if count == 3000:
                  swit = 1
                  flag = 1
                           
            train_writer.add_summary(summary, step)
            train_writer.flush()
            loss_1000_ALL = loss_1000_ALL + G_z_val+ D_z_val+Dice_val
            loss_1000_G = loss_1000_G + G_z_val
            loss_1000_D = loss_1000_D + D_z_val
            loss_1000_Dice = loss_1000_Dice + Dice_val
            loss_echo = loss_echo + G_z_val+ D_z_val+Dice_val

            
            if step % 2000 == 0:
              save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
              logger.info("Model saved in file: %s" % save_path)
              logger.info('-----------Step %d:-------------' % step)
              logger.info('  loss_1000_ALL   : {}'.format(loss_1000_ALL))
              logger.info('  loss_1000_G   : {}'.format(loss_1000_G))
              logger.info('  loss_1000_D   : {}'.format(loss_1000_D))
              logger.info('  loss_1000_Dice   : {}'.format(loss_1000_Dice))
              
              
              loss_1000_G = 0
              loss_1000_ALL = 0
              loss_1000_D = 0
              loss_1000_Dice = 0
              
            if step % num_samples == 0:
              logger.info('------------------------Step %d:---------------------------' % step)
              logger.info('-----------Step %d:-------------' % step)
              logger.info('-----------Step %d:-------------' % step)
              logger.info('!!!!!!!!!!!!  loss_echo   : {}'.format(loss_echo))
              loss_echo = 0
            step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
