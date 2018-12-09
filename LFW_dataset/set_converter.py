#################################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program proccess the lwf_test dataset created by the
# 'setMaker.py' into the final lfw_test dataset.
# It aligns the faces from the pictures. When a picture can not be processed by
# the aligner, the picture is substituted with a previously cropped picture
# downloaded from the 'http://conradsanderson.id.au/lfwcrop/' website
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
hdf5_path = 'dataset_lfw.hdf5'
hdf5_write = tables.open_file(hdf5_path, mode='w')
storage = hdf5_write.create_earray(hdf5_write.root, 'images', img_dtype, shape=data_shape)

# Defining lists needed for the algorithm
y_labels = []
y_storage = []
x_images = []

# Load the dataset made by the 'setMaker.py' program
hdf5_path = 'dataset_lfw_set.hdf5'
hdf5_read = tables.open_file(hdf5_path, mode='r')
data_num = hdf5_read.root.images.shape[0]
prev_label = hdf5_read.root.labels[0]
print("Evaluating %s ."%(hdf5_path))

# Blur limit
blur_limit = 100
blur = 0

# Iterates through the pictures in a dataset.
for i in range(data_num):
	# Reading the image and the corresponding label.
	image = hdf5_read.root.images[i]
	label = hdf5_read.root.labels[i]
	print("Evaluation: %.3f %%"%(float(i)/float(data_num-1)*100),end="\r", flush=True)
	# Detect face and return bounding box
	rect = alignment.getLargestFaceBoundingBox(image)
	# Transform image using specified face landmark indices and crop image to 64x64
	img_aligned = alignment.align(64, image, rect, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    # If the aligning was successful
	if img_aligned is not None:
        # Check if the picture has to much black (empty) pixels, which would indicate a bad image, because
        # a face at the edge of a picture could be recognized and aligned even if it's missing part of the face
		black = 0
        # Counting black (empty pixels)
		for l in range(img_aligned.shape[0]):
			for k in range(img_aligned.shape[1]):
				if img_aligned[l,k] == 0:
					black+=1
        # Checking the threshold for black (empty) pixels
		if black<100:
			# Calculating blur value.
			blur = cv2.Laplacian(img_aligned, cv2.CV_64F).var()
			# Comparing the blur value to the threshold.
			if blur > blur_limit:
				x_images.append(img_aligned[None])
				y_labels.append(label)
			# If the picture was not adequate, we substitute it with a previously cropped image
			else:
				path = 'lfwcrop_grey/faces/' + label.decode("utf-8") + '.pgm'
				image = cv2.imread(path, -1)
				x_images.append(image[None])
				y_labels.append(label)
		# If the picture was not adequate, we substitute it with a previously cropped image
		else:
			path = 'lfwcrop_grey/faces/' + label.decode("utf-8") + '.pgm'
			image = cv2.imread(path, -1)
			x_images.append(image[None])
			y_labels.append(label)
	# If the picture was not adequate, we substitute it with a previously cropped image
	else:
		path = 'lfwcrop_grey/faces/' + label.decode("utf-8") + '.pgm'
		image = cv2.imread(path, -1)
		x_images.append(image[None])
		y_labels.append(label)
	# Store the images and the labels
	for im in x_images:
		storage.append(im)
	y_storage.extend(y_labels)
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