# RAS-NET (TensorFlow)
An implementation of RSA-NET using TensorFlow (work in progress).


## Data path
We upload part of training samples to run our model in '/train/' for demo. 
Complete dHCP and NeoBrains12 can be achieved from each website of database or you can replace with your own data sets.

When you use your own unlabeled data, you need to register the training images and labels of the source domain to your data through 3D linear registration or registration network. Then, RAS-NET domain adaptive transfer can be carried out. For data format, please refer to' x',' y' and' z' in the training example files.(x:label image,y:source domain,z:target domain)


## Environment

* TensorFlow '2.3.1'
* Python 3.9.0
* 


## Training

Here is the list of arguments:
```
usage: train.py [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan]
                [--norm NORM][--DataPath DATAPATH][--valPath VALPATH]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--ngf NGF] [--class_num CLASS_NUM] [--Y Y]
                [--load_model LOAD_MODEL]

optional arguments:
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_size IMAGE_SIZE
                        image size, default: 256
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True
  --nouse_lsgan
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --ngf NGF             number of gen filters in first conv layer, default: 64
  --DataPath DATAPATH   data sets files for training, default:

  --valPath VALPATH     validation file for training, default:

  --load_model LOAD_MODEL
                        folder of saved model that you wish to continue
                        training (e.g. 20170602-1936), default: None
```


## Inference
```
usage: inf_gen.py [--model MODEL] [--model1 MODEL1] [--model2 MODEL2]
			[--image_size IMAGE_SIZE]
			[--valPath VALPATH] 
			[--rsltPath RSLTPATH] 	

optional arguments:
  --model MODEL 
				Target domain Generator Model path of RAS, default: 
  --model1 MODEL1
                        feature Generator Model path, default: 
  --model2 MODEL2
                        segmentation Model path, default: 

  --image_size IMAGE_SIZE
                        image size, default: 400
  --valPath VALPATH     
				validation file for training, default:

  --rsltPath RSLTPATH
                        result images path, default:



usage: inf_seg.py [--model5 MODEL5] [--image_size IMAGE_SIZE]
			[--valPath VALPATH] 
			[--rsltPath RSLTPATH] 	

optional arguments:
  --model5 MODEL5
                        segmentation Model path, default: 

  --image_size IMAGE_SIZE
                        image size, default: 400
  --valPath VALPATH     
				validation file for training, default:

  --rsltPath RSLTPATH
                        result images path, default:


usage: export_gen.py [--model1 MODEL1] [--model2 MODEL2] 
                        [--checkpoint_dir CHECKPOINT_DIR]

optional arguments:

  --model1 MODEL1
                       Generated Target domain Generator Model path of RAS, default:  
  --model2 MODEL2
                       Generated feature Model path of RAS, default:
  --checkpoint_dir CHECKPOINT_DIR
                       Model checkpoint path

usage: export_seg.py [--model MODEL]
                        [--checkpoint_dir CHECKPOINT_DIR]

optional arguments:

  --model MODEL
                       Generated Target domain Generator Model path of RAS, default:  
  --checkpoint_dir CHECKPOINT_DIR
                       Model checkpoint path
```















