##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: This program is the backend for the 'webcam_app-py' program
# detects faces on the web camera feed and decides, if they belong to one
# of the same persons, who was recorded before.
##########################################################################

# General Python imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, shutil, errno, stat
# keras + tensorflow imports necessary for our model
from keras.models import load_model
import keras
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras import optimizers
import tensorflow as tf
# Import the necessary packages for face aligning
import dlib
from align import AlignDlib

# The class for the web camera testing
class WebcamTest():
    def  __init__(self):
        # Initialize the dlib face alignment utility
        self.alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        # Load the model
        self.model = load_model('weights_semihard_02_dec05.hdf5', custom_objects={'triplet_loss': self.triplet_loss, 'tf': tf})

        # Get the web camera feed
        self.cap = cv2.VideoCapture(0)

        # Defining variables 
        self.threshold = 0.7
        self.base_images = []
        self.distances = []
        self.set_new_person = False
        self.saving = False
        self.pressed = 0
        self.next_base_image = 0
        self.names = []
        self.saved_images = []
        self.counter = 0

        # Defining the path for image saving
        self.path = os.getcwd() + '\persons'
        # Delete previous directories
        if os.path.isdir(self.path):
            shutil.rmtree(self.path, onerror=self.onerror)
        # Create the 'persons' directory
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def onerror(self, func, path, exc_info):
        """
        Error handler for `shutil.rmtree`.

        If the error is due to an access error (read only file)
        it attempts to add write permission and then retries.

        If the error is for another reason it re-raises the error.

        Usage : `shutil.rmtree(path, onerror=onerror)`
        """
        import stat
        if not os.access(path, os.W_OK):
            # Is the error an access error ?
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise

    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    def rect_to_bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
 
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    # The model calculates the distance between the two images
    def prediction(self, network, pic_1, pic_2):
        # increase img dimensions
        img1 = np.expand_dims(pic_1, axis=0)
        img1 = np.expand_dims(img1, axis=3)
        img2 = np.expand_dims(pic_2, axis=0)
        img2 = np.expand_dims(img2, axis=3)
        # calculate the network's prediction on img1 and img2
        preds = network.predict([img1, img2, img1])[0]
        pred1 = preds[:128]
        pred2 = preds[128:256]
        # calculate the distance between the two images
        dist = np.sum(np.square(pred1-pred2))
        # print("distance between pictures: {:2.4f}".format(dist))
        return dist

    def triplet_loss(self, y_true, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """

        anchor = y_pred[:,:128]
        positive = y_pred[:,128:256]
        negative = y_pred[:,256:]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        # compute loss
        basic_loss = pos_dist-neg_dist+alpha
        loss = K.maximum(basic_loss,0.0)
    
        return loss

    # Aligns the given image
    def image_processing(self, image):
        # Detect face and return bounding box
        rect = self.alignment.getLargestFaceBoundingBox(image)
        img_aligned = self.alignment.align(64, image, rect, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        return img_aligned

    # Saves the given image with the given name
    def save_pictures(self, image, path, name, extension):
        img_item = path + '\\' + name + "." + extension
        cv2.imwrite(img_item, image)

    # The main part of the class
    def cycle(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        # Turn the image into a greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        rects = self.detector(gray, 1)
        for rect in rects:
            (x, y, w, h) = self.rect_to_bb(rect)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]   

            # Align the image
            aligned_image = self.alignment.align(64, gray, rect, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            
            # If the alignment was succesful
            if aligned_image is not None:
                # If there is a new recording save the recorded images (the saving part is done in the 'webacm_app.py')
                if self.set_new_person:
                    self.counter += 1
                    if self.counter%10 == 0:
                        if len(self.saved_images) < 10:
                            self.saved_images.append(aligned_image)
                        else:
                            self.counter = 0
                # Create a directory for the new person and save the person's images
                if self.saving:
                    self.path_person = self.path + '\\' + self.names[-1]
                    os.mkdir(self.path_person)
                    for (i, picture) in enumerate(self.base_images[-1]):
                        self.save_pictures(picture, self.path_person, self.names[-1] + '_' + str(i), 'jpg')
                    self.saving = False
                # Calculate the average distance from the save distance
                for i in range(len(self.base_images)):
                    average_distance = 0
                    for base_image in self.base_images[i]:
                        average_distance += self.prediction(self.model, aligned_image/255.0, base_image/255.0)
                    self.distances[i] = average_distance/len(self.base_images[i])
            # If the  picture alignment was unsucessful, then the distance belonging to the picture is four
            else:
                for i in range(len(self.distances)):
                    self.distances[i] = 4
            if self.set_new_person:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0) # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            min_index = 0
            # Calculate the minimum distances from all person
            for (i, distance) in enumerate(self.distances):
                if distance < self.distances[min_index]:
                    min_index = i
            # Display the distance values on the feed       
            if len(self.distances)>0 and self.distances[min_index] < self.threshold:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = self.names[min_index] + "{:2.2f}".format(self.distances[min_index])
                color = (255, 255, 255)
                stroke = 1
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 255, 255)
                stroke = 1
                if len(self.distances)>0:
                    name = "{:2.2f}".format(self.distances[min_index])
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        return frame

# test = WebcamTest()
# while True:
#     frame = test.cycle()
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
#     if cv2.waitKey(20) & 0xFF == ord('n') and test.pressed == 0:
#         test.set_new_person = not test.set_new_person
#         print(test.set_new_person)
#         test.pressed = 10
#     if test.pressed > 0:
#         test.pressed -= 1

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()