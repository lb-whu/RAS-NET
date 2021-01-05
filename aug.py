# -*- coding: utf-8 -*-
from skimage import transform as trans
import numpy as np
from scipy.ndimage.interpolation import zoom
def getMeanVarIMG(imgy,imgz,labelx,labelz,label_num):
    label_num = 8
    meany_img = np.zeros_like(imgy)
    vary_img = np.zeros_like(imgy)
    meanz_img = np.zeros_like(imgy)
    varz_img = np.zeros_like(imgy)
    for k in range(0,label_num):
        meanyk = np.sum(imgy*labelx[:,:,k])/(np.sum(labelx[:,:,k])+1)
        varyk = np.sum(np.square(imgy*labelx[:,:,k]-meanyk*labelx[:,:,k]))/(np.sum(labelx[:,:,k])+1)
        varyk = np.sqrt(varyk + 1e-5)
        meany_img = meany_img + meanyk*labelx[:,:,k]
        vary_img = vary_img + varyk*labelx[:,:,k]
    for k in range(0,label_num):
        meanzk = np.sum(imgz*labelz[:,:,k])/(np.sum(labelz[:,:,k])+1)
        varzk = np.sum(np.square(imgz*labelz[:,:,k]-meanzk*labelz[:,:,k]))/(np.sum(labelz[:,:,k])+1)
        varzk = np.sqrt(varzk + 1e-5)
        meanz_img = meanz_img + meanzk*labelx[:,:,k]
        varz_img = varz_img + varzk*labelx[:,:,k]
    meany_img = meany_img.astype(np.float32);meany_img = meany_img[np.newaxis,:,:,np.newaxis]
    vary_img = vary_img.astype(np.float32);vary_img = vary_img[np.newaxis,:,:,np.newaxis]
    meanz_img = meanz_img.astype(np.float32);meanz_img = meanz_img[np.newaxis,:,:,np.newaxis]
    varz_img = varz_img.astype(np.float32);varz_img = varz_img[np.newaxis,:,:,np.newaxis]
    return meany_img,vary_img,meanz_img,varz_img

def getMeanVar(imgf,imgz,labelx,labelz,label_num):
 #   STD = np.std(X.flatten(), axis=0) 
#    MAX = np.min(STD)
#    av = np.average(X.flatten(), axis=0)
#    print('max',MAX)
#    print('av',av)
#    X[STD > 0.8*np.max(STD)] =np.average(X.flatten(), axis=0)+ 0.8*np.max(STD)
    imgf2 = imgf.flatten()
    imgz2 = imgz.flatten()
    labelx2 = labelx.flatten()
    labelz2 = labelz.flatten()
    wh_Mean3 = 0
    wh_var3 = 0
    for k in range(0,label_num):
      if (labelx2.__contains__(k))&(labelz2.__contains__(k)):
        whole = np.delete(imgf2,np.where(labelx2!=k))
        wh_Mean = np.average(whole.flatten(), axis=0)
        wh_var = np.std(whole.flatten(), axis=0)
        
        whole2 = np.delete(imgz2,np.where(labelz2!=k))
        wh_Mean2 = np.average(whole2.flatten(), axis=0)
        wh_var2 = np.std(whole2.flatten(), axis=0)
        
        wh_Mean3 = wh_Mean3 + np.abs(wh_Mean - wh_Mean2)
        wh_var3 = wh_var3 + np.abs(wh_var - wh_var2)
    
    return wh_Mean3,wh_var3
def align_label(slabel,tlabel):
  flabel = tlabel.copy()
  if (np.sum(slabel[:,:,:,1]) !=0) & (np.sum(tlabel[:,:,:,1]) ==0):
    flabel[:,:,:,1] = tlabel[:,:,:,2].copy()+tlabel[:,:,:,4].copy()+tlabel[:,:,:,5].copy()
  if (np.sum(slabel[:,:,:,2]) !=0) & (np.sum(tlabel[:,:,:,2]) ==0):
    flabel[:,:,:,2] = tlabel[:,:,:,1].copy()+tlabel[:,:,:,4].copy()+tlabel[:,:,:,5].copy()
  if (np.sum(slabel[:,:,:,3]) !=0) & (np.sum(tlabel[:,:,:,3]) ==0):
    flabel[:,:,:,3] = 1-tlabel[:,:,:,0].copy()
  if (np.sum(slabel[:,:,:,4]) !=0) & (np.sum(tlabel[:,:,:,4]) ==0):
    flabel[:,:,:,4] = tlabel[:,:,:,1].copy()+tlabel[:,:,:,2].copy()+tlabel[:,:,:,5].copy()
  if (np.sum(slabel[:,:,:,5]) !=0) & (np.sum(tlabel[:,:,:,5]) ==0):
    flabel[:,:,:,5] = tlabel[:,:,:,1].copy()+tlabel[:,:,:,2].copy()+tlabel[:,:,:,4].copy()
  if (np.sum(slabel[:,:,:,6]) !=0) & (np.sum(tlabel[:,:,:,6]) ==0):
    if (np.sum(tlabel[:,:,:,7])) ==0: 
      flabel[:,:,:,6] = 1-tlabel[:,:,:,0].copy()
    else:
      flabel[:,:,:,6] = tlabel[:,:,:,7].copy()
  if (np.sum(slabel[:,:,:,7]) !=0) & (np.sum(tlabel[:,:,:,7]) ==0):
    if (np.sum(tlabel[:,:,:,6])) ==0: 
      flabel[:,:,:,7] = 1-tlabel[:,:,:,0].copy()
    else:
      flabel[:,:,:,7] = tlabel[:,:,:,6].copy()
  return flabel
  

def isfinish(csum,cmean,cvar):
 #   STD = np.std(X.flatten(), axis=0) 
#    MAX = np.min(STD)
#    av = np.average(X.flatten(), axis=0)
#    print('max',MAX)
#    print('av',av)
#    X[STD > 0.8*np.max(STD)] =np.average(X.flatten(), axis=0)+ 0.8*np.max(STD)
    if (csum<1000)&(cvar<1.1):
      return True
    else:
      return False
        
def crop(x,cpsize):
    D,W,H,f = x.shape
#    print('W',x.shape)
    for i in range(0,D):
        if np.max(x[i,:,:,:]) > 0:
            ind_ld = i
            break
    for i in range(D-1,-1,-1):
        if np.max(x[i,:,:,:]) > 0:
            ind_rd = i + 1
            break
    for i in range(0,W):
        if np.max(x[:,i,:,:]) > 0:
            ind_lw = i
            break
    for i in range(W-1,-1,-1):
        if np.max(x[:,i,:,:]) > 0:
            ind_rw = i + 1
            break
    for i in range(0,H):
        if np.max(x[:,:,i,:]) > 0:
            ind_lh = i
            break
    for i in range(H-1,-1,-1):
        if np.max(x[:,:,i,:]) > 0:
            ind_rh = i + 1
            break
    d = ind_rd - ind_ld;w = ind_rw - ind_lw;h = ind_rh - ind_lh
    box = np.zeros((6),dtype = np.int32)
    if cpsize[0] == 0:
        box[0] = ind_ld; box[1] = ind_rd; box[2] = ind_lw; box[3]=ind_rw; box[4]=ind_lh; box[5]=ind_rh
    else:
        box[0] = np.max([ind_ld - np.round((cpsize[0]-d)/2),0])
        box[2] = np.max([ind_lw - np.round((cpsize[1]-w)/2),0])
        box[4] = np.max([ind_lh - np.round((cpsize[2]-h)/2),0])
        box[1] = box[0] + cpsize[0]
        box[3] = box[2] + cpsize[1]
        box[5] = box[4] + cpsize[2]
#    box = [ind_ld,ind_rd,ind_lw,ind_rw,ind_lh,ind_rh]
    box = box.astype(np.int32)

    return box,box[1]-box[0],box[3]-box[2],box[5]-box[4]
def cropback(x,box,origsize):
    img = np.zeros((origsize[0],origsize[1]),dtype = np.float32)
    img[box[2]:box[3],box[4]:box[5]] = x
    return img
def zm(x):

    datax = zoom(x, zoom = [1,0.5,0.5,1], order=0)
    return datax
def zm_single(x,p_size):
    D,W,H,f = x.shape
    factorD = p_size/D
    factorW = p_size/W
    factorH = p_size/H
    datax = zoom(x, zoom = [factorD,factorW,factorH,1], order=0)
    return datax
def zm_back(x,y,p_size):
    D,W,H,f = x.shape
    factorD = D/p_size
    factorW = W/p_size
    factorH = H/p_size
    datay = zoom(y, zoom = [factorD,factorW,factorH,1], order=0)
    return datay
def fixbox(ind,p_size,box2size):
    box = np.zeros((6),dtype = np.float32)
    if box2size[0]*box2size[3]/p_size < p_size-8:    #d
        gap = np.round((p_size-box2size[0]*box2size[3]/p_size)/2)
        box[0] = np.max([np.round(ind[0]*box2size[3]/p_size)-gap,0]);box[1] = np.min([np.round(ind[1]*box2size[3]/p_size)+gap,box2size[3]]) 
    else:
        box[0] = np.max([np.round(ind[0]*box2size[3]/p_size)-16,0]);box[1] = np.min([np.round(ind[1]*box2size[3]/p_size)+16,box2size[3]]) 
        
    if box2size[1]*box2size[4]/p_size < p_size-8:    #d
        gap = np.round((p_size-box2size[1]*box2size[4]/p_size)/2)
        box[2] = np.max([np.round(ind[2]*box2size[4]/p_size)-gap,0]);box[3] = np.min([np.round(ind[3]*box2size[4]/p_size)+gap,box2size[4]-6]) 
    else:
        box[2] = np.max([np.round(ind[2]*box2size[4]/p_size)-16,0]);box[3] = np.min([np.round(ind[3]*box2size[4]/p_size)+16,box2size[4]-6]) 

    if box2size[2]*box2size[5]/p_size < p_size-8:    #d
        gap = np.round((p_size-box2size[2]*box2size[5]/p_size)/2)
        box[4] = np.max([np.round(ind[4]*box2size[5]/p_size)-gap,0]);box[5] = np.min([np.round(ind[5]*box2size[5]/p_size)+gap,box2size[5]-6]) 
    else:
        box[4] = np.max([np.round(ind[4]*box2size[5]/p_size)-16,0]);box[5] = np.min([np.round(ind[5]*box2size[5]/p_size)+16,box2size[5]-6]) 
    box = box.astype(np.int32)
    return box  
def fixfeature(ind,f_size,boxsize):
    box = np.zeros((6),dtype = np.float32)
    box[0] = np.round(ind[0] * f_size / boxsize[0])
    box[1] = np.round(ind[1] * f_size / boxsize[0])
    box[2] = np.round(ind[2] * f_size / boxsize[1])
    box[3] = np.round(ind[3] * f_size / boxsize[1])
    box[4] = np.round(ind[4] * f_size / boxsize[2])
    box[5] = np.round(ind[5] * f_size / boxsize[2])
    box = box.astype(np.int32)
    return box
def aug(x,y,label_num):
    """
  aug images from input_dir
  Args:
    flag: 0--current 
  Returns:
    auged x
    """
    flag = np.random.randint(low = 0,high = 3, size=(1))
    rot = np.random.randint(low = -10,high = 11, size=(1))
    label = np.zeros((y.shape[0],y.shape[1],y.shape[2],label_num),dtype = np.int32)
    for k in range(0,label_num):
        gt = y.copy()
        gt[gt == k] = 100
        gt[gt != 100] = 0
        label[:,:,:,k] =  gt           
    if flag == 0:
        img2 = np.reshape(x.copy(),(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        l2 = np.reshape(label.copy(),(label.shape[0],label.shape[1],label.shape[2]*label.shape[3]))
        datax = trans.rotate(img2, rot)
        datay = trans.rotate(l2, rot)
        datax = np.reshape(datax,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        datay = np.reshape(datay,(label.shape[0],label.shape[1],label.shape[2],label.shape[3]))
        if np.random.randint(low = 0,high = 2, size=(1)) == 1:
            datax = datax.transpose((1,2,0,3))
            datay = datay.transpose((1,2,0,3))
            datax2 = np.reshape(datax.copy(),(datax.shape[0],datax.shape[1],datax.shape[2]*datax.shape[3]))
            datay2 = np.reshape(datay.copy(),(datay.shape[0],datay.shape[1],datay.shape[2]*datay.shape[3]))        
            datax2 = np.fliplr(datax2)
            datay2 = np.fliplr(datay2)
            datax2 = np.reshape(datax2,(datax.shape[0],datax.shape[1],datax.shape[2],datax.shape[3]))
            datay2 = np.reshape(datay2,(datay.shape[0],datay.shape[1],datay.shape[2],datay.shape[3])) 
            datax = datax2.transpose((2,0,1,3))
            datay = datay2.transpose((2,0,1,3))
    else:
        if flag == 1:
            x = x.transpose((1,2,0,3)).copy()
            label = label.transpose((1,2,0,3)).copy()
        elif flag == 2:
            x = x.transpose((2,0,1,3)).copy()
            label = label.transpose((2,0,1,3)).copy()
        img2 = np.reshape(x.copy(),(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        l2 = np.reshape(label.copy(),(label.shape[0],label.shape[1],label.shape[2]*label.shape[3]))
        datax = trans.rotate(img2, rot)
        datay = trans.rotate(l2, rot)

        datax = np.reshape(datax,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        datay = np.reshape(datay,(label.shape[0],label.shape[1],label.shape[2],label.shape[3]))
        if flag == 1:
            if np.random.randint(low = 0,high = 2, size=(1)) == 1:
                datax = np.fliplr(datax)
                datay = np.fliplr(datay)
            datax = datax.transpose((2,0,1,3))
            datay = datay.transpose((2,0,1,3))
        elif flag == 2:
            datax = datax.transpose((1,2,0,3))
            datay = datay.transpose((1,2,0,3))
    ind = np.argmax(datay,axis = 3)
    ind = ind.astype(np.int32)
    label = np.zeros((y.shape[0],y.shape[1],y.shape[2],label_num),dtype = np.int32)
    for k in range(0,label_num):
        gt = ind.copy()
        gt[gt == k] = 10
        gt[gt != 10] = 0
        gt[gt == 10] = 1
        label[:,:,:,k] =  gt
    return datax,label
def L2HOT(y,label_num):
    label = np.zeros((y.shape[0],y.shape[1],label_num),dtype = np.int32)
    for k in range(0,label_num):
        gt = y.copy()
        gt[gt == k] = 10
        gt[gt != 10] = 0
        gt[gt == 10] = 1
        label[:,:,k] =  gt
    return label  
def aug2D(x,y,z,label_num):
    """
  aug images from input_dir
  Args:
    flag: 0--current 
  Returns:
    auged x
    """
    flag = np.random.randint(low = 0,high = 2, size=(1))
    rot = np.random.randint(low = -10,high = 11, size=(1))
    label = np.zeros((y.shape[0],y.shape[1],label_num),dtype = np.int32)
    for k in range(0,label_num):
        gt = x.copy()
        gt[gt == k] = 100
        gt[gt != 100] = 0
        label[:,:,k] =  gt
           
    if flag == 0:

        datax = trans.rotate(label, rot)
        datay = trans.rotate(y, rot)
        dataz = trans.rotate(z, rot)

  
        datax = np.fliplr(datax)
        datay = np.fliplr(datay)
        dataz = np.fliplr(dataz)
    else:
        datax = trans.rotate(label, rot)
        datay = trans.rotate(y, rot)
        dataz = trans.rotate(z, rot)
        
    ind = np.argmax(datax,axis = 2)
    ind = ind.astype(np.int32)
    label = np.zeros((y.shape[0],y.shape[1],label_num),dtype = np.int32)
    for k in range(0,label_num):
        gt = ind.copy()
        gt[gt == k] = 10
        gt[gt != 10] = 0
        gt[gt == 10] = 1
        label[:,:,k] =  gt
    return label,datay,dataz