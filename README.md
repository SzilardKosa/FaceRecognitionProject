# FaceRecognitionProject
- Team name: LearnFromYourMistakes
- Team members: Lajos Bodo, Szilard Kosa
- Emails: ifj.bodo.lajos@gmail.com , kszilard777@gmail.com

## Project goals

Our goal is to implement [FaceNet](https://arxiv.org/abs/1503.03832) network in keras. 

In our solution, the input of the network is a fixed size, grayscale picture of a person. The output is a 128 dimensional embedding on a unit hypersphere. The relation between two pictures can be determined from the distance of their embeddings. If two embeddings are close to each other that means the persons on the pictures look similar.
![pic1](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2i21SzZWa41AQRl8W64vmcFJ6RknloSflSkN-DtYxtMAWoFSN)

The optimization of the network is done by a triple loss function according to the [FaceNet](https://arxiv.org/abs/1503.03832) documentation. The triplets consists of three pictures, where the first two are from the same person and the third one is from a different person. We are using the [online triplet mining](https://omoindrot.github.io/triplet-loss) technique to create these triplets. Which means the training data consists of batches containing a defined amount of pictures for the defined number of people. The triplets are created only after their embeddings are calutated. 

![pic2](https://qph.fs.quoracdn.net/main-qimg-17cd47a61fa2e0472d569040aacdf2fc)

For test and validation data we use the VGGFace2 dataset. The downloaded images are processed so could be used as proper training images. The procession includes face detection, text removal, face alignment, blurrines and empty pixel filtering. For testing we use the Labeled Faces in the Wild (LFW) dataset. We process the test images in the same way as the training images.

We visualize the training results with a UMAP projection and a python application which recognizes people from a laptop's camera's live feed. The recogniton is being done with our trained model.

## Downloads for the programs
All downloads can be find in the following drive folder: [download](https://drive.google.com/drive/folders/1BvybDG_vqE5Q6wxaai8FkZRRPhEUm780?usp=sharing)

'dataset.hdf5' 

'dataset_lfw.hdf5' 

LFW cropped (Zip) it could be also download from the 'http://conradsanderson.id.au/lfwcrop/' website

'dataset_umap.hdf5'

Our final model: 'weights_final.hdf5'

'frozen_east_text_detection.pb'

'shape_predictor_68_face_landmarks.dat'

'align.py'

'pairs.txt'

'overlap.txt'

'set_index.txt'

## Short description of the programs

### Programs to process the VGGFace2 dataset
#### pictures_to_hdf5.py
The program goes through the unzipped VGGFace2 dataset, scales down the pictures to 128x128 size and turns them into grayscale, and saves the results into a HDF5 file. It is a very time consuming process. Becuse of the size of the database, the converting was done in three steps resulting in three HDF5 files. (Before running this program the database must be unzipped in a folder called 'test' next to the program.)
#### dataset_analizer.py
The program goes through the previously created files and counts the number of images for each person. The number is determined from the number of labels. The result can be seen in Data_Analysis.xlsx .
#### dataset_text_remover
The program process and merges the datasets created by the 'pictures_to_hdf5.py' into one dataset ('dataset_without_text.hdf5'). It filters out the pictures with text on them. It needs the *'frozen_east_text_detection.pb'* file for the text detection.
#### dataset_filter.py
The program process the 'dataset_without_text.hdf5' file created by the 'dataset_text_remover.py' into the final dataset. It crops the faces from the pictures, saves only those persons who has at least a predefined number of pictures. The pictures that fails the given blurry limit are not taken into account. For the face alignment it imports the *align.py* program. For the face detection it needs the *'shape_predictor_68_face_landmarks.dat'* file.
#### dataset_viewer.ipynb
The notebook displays the pictures in the dataset. The notebook loads the chosen dataset from the same directory where this program is located.

### Programs to process the LFW dataset
#### set_maker.py
The program creates a HDF5 dataset ('dataset_lfw_set.hdf5') from the LFW dataset. The dataset's structure is defined by the *'pairs.txt'* file. The program also needs *'the overlap.txt'* file to filter out the person who are also present in the VGGFace2 dataset. The program creates a text file, named 'set_index.txt', which contains info for each set's place in the dataset file. It is needed for the lfw_test.ipynb.
There is no modification on the images.
#### set_converter.py
The program proccess the 'dataset_lfw_set.hdf5' created by the 'set_maker.py' into the final 'dataset_lfw.hdf5' dataset file. It aligns the faces from the pictures. When a picture could not be processed by the aligner, the picture is substituted with a previously cropped picture from the unzipped LFW Cropped dataset. For the face alignment it imports the *align.py* program. For the face detection it needs the *'shape_predictor_68_face_landmarks.dat'* file.

### Programs for the training and testing
#### trainer.ipynb
The main program. It loads the *'dataset.hdf5'* for the training and the *'dataset_umap.hdf5'* for the visualization from the mounted drive. The model is defined in the notebook. Then it trains the model on the dataset and saves it to a specified location in the mounted drive.
#### lfw_test.ipynb
This notebook is used to test our model's performance on the 'dataset_lfw.hdf5' test file. It loads the *'dataset_lfw.hdf5'*, the *'set_index.txt'* and the *chosen model* from the mounted drive. The result shows the model's precision, recall and accuracy ratings. It also generates an ROC curve for the model and calculates the AUC value.

### Programs for the web camera application
#### webcam_app.py
The main program for the web camera application. Run this to start the application. The program needs the *'webcam_test.py'*, the *'webcam.kv'*, the *'align.py'*, the *'shape_predictor_68_face_landmarks.dat'* and the *chosen model* in the same directory. The kivy module also needed for the graphical interface.

How to use the application: After starting the application wait until the model is loaded (usually around 30 sec). Record a person pressing the 'start recording' button. Important: When recording only one person should be in the camera's field of view. Wait until at least 1 picture is recorded (The suggested number is 10, which is also the maximum), then press the same button (This time it reads 'Stop recording'). If the name of the person was different from any other recorded person then the recording is completed. After this the application writes out the recorded name  if it recognises someone on the person's frame. There is a predetermined threshold with the value of 0.7, if the calculated value from the model is smaller than the threshold the person is recognized.
#### webcam_test.py
The backend of the 'webcam_app.py'. It does not work on it's own, because it needs input from the 'webcam_app.py'. The threshold value could be changed here, in the WebcamTest class' constructor.
#### webcam.kv
The graphical layout for the app is defined in this file.

### Other programs
#### triplet_loss.ipynb
This program test the triplet loss method on the MNIST database.

