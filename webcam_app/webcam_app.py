##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program opens an application, in which the webcamera of
# a laptop can be used for live testing the model.
##########################################################################

# Kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# General Python imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

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

# Import the WebcamTest class from the 'webcam_test.py' file, which handles the face recognition
from webcam_test import WebcamTest

# The main app of the program
class WebcamApp(App):
    def build(self):
        return Base()
        
    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()
        
# The base widget, it handles the interactions with the other widgets
class Base(Widget):
    def __init__(self):
        super(Base, self).__init__()
        # Delaring the event of the recording
        self.ids.record_button.bind(on_release = self.recording)

        # Starting the webcamera
        self.webCam = WebcamTest()
        self.ids.cam.start(30, self.webCam, self)

    # Defining the recording event
    def recording(self, instance):
        if self.ids.record_button.text == 'Start recording':
            self.ids.record_button.text = "Stop recording"
        else:
            # If there is no new name given or no image was recorded,
            # the interaction with the 'webcam_test.py' is skipped
            if self.ids.input.text not in self.webCam.names and len(self.webCam.saved_images)>0:
                self.webCam.names.append(self.ids.input.text)
                self.webCam.base_images.append(self.webCam.saved_images)
                self.webCam.saved_images = []
                self.webCam.distances.append(0)
                self.webCam.saving = True
                self.refresh(self.webCam.names)            
            self.webCam.saved_images = []
            self.ids.record_button.text = 'Start recording'
        self.webCam.set_new_person = not self.webCam.set_new_person
    
    # Refreshing the recorded names
    def refresh(self,tomb):
        self.ids.name_box.add_widget(Label(text = tomb[-1], color=(0,0,0,1)))

class Input_text(TextInput):
    pass

class Record(Button):
    pass

# The widget for the web camera display
class Picture(Image):
    def start(self, fps, cam, base, **kwargs):
        self.webCam = cam
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.base = base

    def update(self, dt):
        # Getting the image from the 'webcam_test.py' to diplay it
        frame = self.webCam.cycle();

        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture
        # if len(self.base.webCam.saved_images) > 0:
        #     self.base.ids.label.text = str(len(self.base.webCam.saved_images))
        self.base.ids.record_label.text = "Number of pictures recorded: " + str(len(self.base.webCam.saved_images))

# The main 
if __name__ == '__main__':
    WebcamApp().run()
