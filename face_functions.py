import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import cv2
from numpy import genfromtxt
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')
np.set_printoptions(threshold=sys.maxsize)

#provides 128 dim embeddings for face
def img_to_encoding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input (160x160)
    img = cv2.resize(img, (160, 160))
    
    # Normalize and prepare for model
    img = np.array(img, dtype=np.float32) / 255.0
    x_train = np.expand_dims(img, axis=0)

    #facial embedding from trained model
    embedding = model.predict_on_batch(x_train)
    return embedding

#calculates triplet loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # triplet loss formula 
    pos_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[1])) )
    neg_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[2])) )
    basic_loss = pos_dist - neg_dist + alpha
    
    loss = tf.maximum(basic_loss, 0.0)
   
    return loss

# load the model
model = load_model('facenet_model/model.h5', custom_objects={'triplet_loss': triplet_loss})