# FaceRecognitionProject
- Team name: LearnFromYourMistakes
- Team members: Lajos Bodo, Szilard Kosa
- Emails: ifj.bodo.lajos@gmail.com , kszilard777@gmail.com

## Project goals

Our goal is to implement [FaceNet](https://arxiv.org/abs/1503.03832) network in keras. 

In our solution, the input of the network is a fixed size, grayscale picture of a person. The output is a 128 dimensional embedding on a unit hpersphere. The relation between two pictures can be determined from the distance of their embeddings. If two embeddings are close to each other that means the persons on the pictures look similar.
![pic1](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2i21SzZWa41AQRl8W64vmcFJ6RknloSflSkN-DtYxtMAWoFSN)

The optimization of the network is done by a triple loss function according to the [FaceNet](https://arxiv.org/abs/1503.03832) documentation. The triplets consists of three pictures, where the first two are from the same person and the third one is from a different person. We are using the [online triplet mining](https://omoindrot.github.io/triplet-loss) technique to create these triplets. Which means the training data consists of batches containing a defined amount of pictures for the defined number of people. And the triplets are created only after their embeddings are calutated. 

![pic2](https://qph.fs.quoracdn.net/main-qimg-17cd47a61fa2e0472d569040aacdf2fc)

For test and validation we split the datasets in half, in the first half we check the performance of the network on different pictures from the same person, and in the second half we inspect the nework on pictures from different persons.

## Description of the programs

We used the first three programs in the early state of the database preprocessing. You don't need to run them to be able to run data_preprocessor.py . They are here only to demonstrate how the data preprocessing was done.
#### picutres_to_hdf5.py
The program goes through the unzipped VGGFace2 database, scales down the pictures to 128x128 size and turns them into grayscale, and saves the results into a HDF5 file. It is a very time consuming process. Becuse of the size of the database, the converting was done in three steps resulting in three HDF5 files. (Before running this program the database must be unzipped in a folder called 'test' next to the program.)
#### dataset_analizer.py
The program goes through the previously created files and counts the number of images for each person. The number is determined from the number of labels. The result can be seen in Data_Analysis.xlsx .
#### dataset_merger.py
The program proccess and merges the datasets created by the 'pictures_to_hdf5.py' into the final dataset. It crops the faces from the pictures, saves only those persons who has at least a predefined number of pictures. The pictures that fails the given blurry limit are not taken into account. To download the final data set click [here](https://drive.google.com/drive/folders/17J0BbO4FZ3EHOnPXdc-9_iJDYND8Hs8v?usp=sharing).
#### data_preprocessor.py
The program opens the dataset and preprocesses it. To run it you have to [download](https://drive.google.com/drive/folders/17J0BbO4FZ3EHOnPXdc-9_iJDYND8Hs8v?usp=sharing) the dataset in the same directory where this program is located.
