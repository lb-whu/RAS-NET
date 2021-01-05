import numpy as np
#from hausdorff import hausdorff_distance
#from hausdorff import hausdorff_distance
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import directed_hausdorff
import cv2
def dice_score_list(label_gt, label_pred, n_class):
    """
    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)



def dice_score(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores
    
def dice_score2(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_pred[l_pred==class_id]=100
        img_A = np.array(l_gt == 100, dtype=np.float32).flatten()
        img_B = np.array(l_pred == 100, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores
def sensitivity(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_pred[l_pred==class_id]=100
        img_A = np.array(l_gt == 100, dtype=np.float32).flatten()
        img_B = np.array(l_pred == 100, dtype=np.float32).flatten()
        score = np.sum(img_A * img_B) / (np.sum(img_A)  + epsilon)
        dice_scores[class_id] = score

    return dice_scores
def haff(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_gt[l_gt!=100]=0
        l_gt[l_gt==100]=1
        l_pred[l_pred==class_id]=100
        l_pred[l_pred!=100]=0
        l_pred[l_pred==100]=1
        l_gt = np.reshape(l_gt,(l_gt.shape[0]*l_gt.shape[1],l_gt.shape[2]))
        l_pred = np.reshape(l_pred,(l_pred.shape[0]*l_pred.shape[1],l_pred.shape[2]))
        dist = directed_hausdorff(l_gt, l_pred)
#        print(dist[0])
        scores[class_id] = dist[0]

    return scores    
def haff2(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_gt[l_gt!=100]=0
        l_gt[l_gt==100]=1
        l_pred[l_pred==class_id]=100
        l_pred[l_pred!=100]=0
        l_pred[l_pred==100]=1
#        l_gt.flatten()
#        l_pred.flatten()
        l_gt = l_gt.astype(np.uint8)
        l_pred = l_pred.astype(np.uint8)
#        l_gt = np.reshape(l_gt,(l_gt.shape[0]*l_gt.shape[2],l_gt.shape[1]))
#        l_pred = np.reshape(l_pred,(l_pred.shape[0]*l_pred.shape[2],l_pred.shape[1]))
        print('l_gt.shape:',l_gt.shape)
        print('l_pred.shape:',l_pred.shape)
        count = 0
        dist = 0
        c1 = np.zeros((l_gt.shape[0]*l_gt.shape[1],l_gt.shape[2]),dtype = np.uint8)
        c2 = np.zeros((l_pred.shape[0]*l_pred.shape[1],l_pred.shape[2]),dtype = np.uint8)
        for i in range(50):
          c1[i*50:(i*50+400),:] = l_gt[i,:,:].copy()
          c2[i*50:(i*50+400),:] = l_pred[i,:,:].copy()
#          if (np.sum(c1)!=0)&(np.sum(c2)!=0):
        
        contours1, hierarchy = cv2.findContours(c1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  
        contours2, hierarchy = cv2.findContours(c2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
        print('contours1 len:',len(contours1)) 
        print('contours2 len:',len(contours2)) 
        cnt1 = contours1[0]
        cnt2 = contours2[0]
        dist = hausdorff_sd.computeDistance(cnt1, cnt2)
#        if count == 0:
#          dist = hausdorff_sd.computeDistance(cnt1, cnt2)
#        else:
#          dist = (dist + hausdorff_sd.computeDistance(cnt1, cnt2))/2
        print(str(class_id)+':',dist)

        scores[class_id] = dist

    return scores
# Hausdorff Distance
def HausdorffDist(label_gt,label_pred,n_class):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_gt[l_gt!=100]=0
        l_gt[l_gt==100]=1
        l_pred[l_pred==class_id]=100
        l_pred[l_pred!=100]=0
        l_pred[l_pred==100]=1
        print('~~')
        D_mat = np.sqrt(inner1d(l_gt,l_gt)[np.newaxis].T + inner1d(l_pred,l_pred)-2*(np.dot(l_gt,l_pred.T)))
        # Find DH
        print(D_mat)
        dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
        print(dH)
    scores[class_id] = dH    
    return(dH)

def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
