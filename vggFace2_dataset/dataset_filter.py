#################################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program process the dataset created by the
# 'dataset_text_remover.py' into the final dataset.
# It crops the faces from the pictures, saves only those persons who has
# at least a predefined number of pictures. The pictures that fails the
# given blurry limit are not taken into account.
#################################################################################

# This library let's us work with hdf5 format files.
import tables
# Used for image processing and face detection.
import cv2

# Import the necessary packages for face aligning
import dlib
from align import AlignDlib

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

# Creating the HDF5 file.
img_dtype = tables.UInt8Atom()
data_shape = (0, 64, 64)
hdf5_path = 'dataset.hdf5'
hdf5_write = tables.open_file(hdf5_path, mode='w')
storage = hdf5_write.create_earray(hdf5_write.root, 'images', img_dtype, shape=data_shape)

# Defining lists needed for the algorithm
y_labels = []
y_storage = []
x_images = []
# The predefined number of pictures for each person saved into the dataset.
num_limit = 40
num = 0
# The threshold for the calculated blur values.
blur_limit = 500
blur = 0

# The names of the datasets created from the VGGFace2 database.
hdf5_path = 'dataset_without_text.hdf5'

hdf5_read = tables.open_file(hdf5_path, mode='r')
data_num = hdf5_read.root.images.shape[0]
prev_label = hdf5_read.root.labels[0]
print("Evaluating %s ."%(hdf5_path))
# Iterates through the pictures in a dataset.
for i in range(data_num):
	# Reading the image and the corresponding label.
	image = hdf5_read.root.images[i]
	label = hdf5_read.root.labels[i]
	print("Evaluation: %.3f %%"%(float(i)/float(data_num-1)*100),end="\r", flush=True)
	# Evaluating the images of a person.
	if label != prev_label:
		if num == num_limit:
			for im in x_images:
				storage.append(im)
			y_storage.extend(y_labels)
		num = 0
		y_labels = []
		x_images = []
	# If there are more pictures than the defined limit for a person, the following pictures are not evaluated.
	if num < num_limit:
		# Detect face and return bounding box
		rect = alignment.getLargestFaceBoundingBox(image)

		# Transform image using specified face landmark indices and crop image to 64x64
		img_aligned = alignment.align(64, image, rect, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		if img_aligned is not None:
			black = 0
			for l in range(img_aligned.shape[0]):
				for k in range(img_aligned.shape[1]):
					if img_aligned[l,k] == 0:
						black+=1
			if black<100:
				# Calculating blur value.
				blur = cv2.Laplacian(img_aligned, cv2.CV_64F).var()
				# Comparing the blur value to the threshold.
				if blur > blur_limit:
					x_images.append(img_aligned[None])
					y_labels.append(label)
					num += 1
	prev_label = label
# Evaluating the last person.
if num == num_limit:
	for im in x_images:
		storage.append(im)
	y_storage.extend(y_labels)
num = 0
y_labels = []
x_images = []
hdf5_read.close()
print()
print("Closing %s ."%(hdf5_path))

# Saving the labels in the HDF5 file.
hdf5_write.create_array(hdf5_write.root, 'labels', y_storage)
# Closing the file.
hdf5_write.close()
print("HDF5 file closed.")