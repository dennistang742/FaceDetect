"""
Demo for the face detection. Runs the sliding window detector on demo.jpg
Displays the input image, the localized faces and the sliding window mask

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import FaceFinder
from FaceFinder import build_net
import cv2
import sys

import tensorflow as tf
import tfac
import face_ds
import numpy as np


ori_width = 300
ori_height = 300

vcap = cv2.VideoCapture(0)
vcap.set(cv2.CAP_PROP_FPS, 60)
stop_vcap = False

model_path = 'face_model'
#img = cv2.imread("demo.jpg",0)

sess = tfac.start_sess()
y,x_hold,y_hold,keep_prob = build_net(sess)
saver = tf.train.Saver()
saver.restore(sess,model_path)

while( not stop_vcap):
    ret, ori_img_bgr = vcap.read()
    ori_img_bgr = cv2.resize(ori_img_bgr, (ori_width, ori_height))
    ori_img_gray = cv2.cvtColor(ori_img_bgr, cv2.COLOR_BGR2GRAY)
    faces, mask = FaceFinder.localize( ori_img_gray, y, sess, x_hold, keep_prob)
    cv2.imshow("faces",faces)
    cv2.imshow("sliding window mask",mask)
    cv2.imshow("input image", (ori_img_gray+faces).astype(np.uint8))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_vcap = True
        break
    #sys.exit()

sess.close()
vcap.release()
cv2.destroyAllWindows()
