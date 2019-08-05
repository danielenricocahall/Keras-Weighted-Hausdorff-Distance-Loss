'''
Created on Oct 12, 2018

@author: daniel
'''
import keras.backend as K
import math

def dice_coef(y_true, y_pred, smooth=1e-3):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
    

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def dice_coef_multilabel(M):
    def loss(y_true, y_pred):
        dice = 0
        for index in range(M):
            dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return dice / M
    return loss



def dice_coef_multilabel_loss(y_true, y_pred):
    return -K.log(dice_coef_multilabel(y_true, y_pred))


## Based on the loss described in "Prostate Segmentation using 2D Bridged U-net" (https://arxiv.org/pdf/1807.04459.pdf)
def dice_coef_multilabel_cos_loss(M, Q = 2):
    def loss(y_true, y_pred):
        dice = 0
        for index in range(M):
            DSC = dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
            dice += K.pow(K.cos( (math.pi / 2) * DSC), Q)
    return loss



    





