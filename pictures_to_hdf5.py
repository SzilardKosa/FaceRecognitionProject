##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program goes through the unzipped VGGFace2 database,
# scales down the pictures to 128x128 size and turns them
# into grayscale, and saves the results into a HDF5 file.
##########################################################################

# We use this module to go through the files.
import os
# Used for image processing and face detection.
import cv2
# This library let's us work with hdf5 format files.
import tables

# The base of the face detection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Defining the folder, where the pictures are stored.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "test")

# Creating the HDF5 file.
img_dtype = tables.UInt8Atom()
data_shape = (0, 128, 128)
hdf5_path = 'dataset_04.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='w')
storage = hdf5_file.create_earray(hdf5_file.root, 'images', img_dtype, shape=data_shape)


y_labels = []
x_train = []

# Iterates through the folders.
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			# The label is created from the folder's numbering.
			label = os.path.basename(root).replace("n", "").lower()
			# Reading the image.
			im = cv2.imread(path)
			# Resizing the image.
			im = cv2.resize(im,(128,128))
			# Turning the image into grayscale.
			gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			# The picture is saved in the HDF5 file, for every face the program detects.
			# In optimal case it's only one time.
			faces = face_cascade.detectMultiScale(gray, 1.3, 6)
			for(x, y, w, h) in faces:
				y_labels.append(int(label))
				storage.append(gray[None])
				print("Label: %i"%(int(label)))

# Saving the labels in the HDF5 file.
hdf5_file.create_array(hdf5_file.root, 'labels', y_labels)
# Closing the file.
hdf5_file.close()
print("HDF5 file closed.")
