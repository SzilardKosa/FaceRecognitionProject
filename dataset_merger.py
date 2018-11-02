#################################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program proccess and merges the datasets created by the
# 'pictures_to_hdf5.py' into the final dataset.
# It crops the faces from the pictures, saves only those persons who has
# at least a predefined number of pictures. The pictures that fails the
# given blurry limit are not taken into account.
#
# The face alignment is based on the following example:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
#################################################################################

# This library let's us work with hdf5 format files.
import tables
# Used for image processing and face detection.
import cv2

# Import the necessary packages for face aligning
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib

# For text detection
from imutils.object_detection import non_max_suppression

def text_detector(image, min_confidence, net, width, height):
	# grab the image dimensions
	(H, W) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (width, height)
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# grab the number of rows and columns from the scores volume
	(numRows, numCols) = scores.shape[2:4]
	
	picture_with_text = 0
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities)
		scoresData = scores[0, 0, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < min_confidence:
				continue  
			else:
				picture_with_text = 1
	return picture_with_text

# Creating the HDF5 file.
img_dtype = tables.UInt8Atom()
data_shape = (0, 64, 64)
hdf5_path = 'dataset.hdf5'
hdf5_write = tables.open_file(hdf5_path, mode='w')
storage = hdf5_write.create_earray(hdf5_write.root, 'images', img_dtype, shape=data_shape)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=64)

# load the pre-trained EAST text detector
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

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
hdf5_paths = ['dataset_01.hdf5','dataset_02.hdf5','dataset_03.hdf5']

# Iterates through the datasets.
for hdf5_path in hdf5_paths:
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
		# Check text
		color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		if text_detector(color, 0.3, net, 128, 128) < 1:
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
				# detect faces in the grayscale
				rects = detector(image, 2)
				# loop over the face detections
				for rect in rects:
					# extract the ROI of the *original* face, then align the face
					# using facial landmarks
					(x, y, w, h) = rect_to_bb(rect)
					if x>0 and y>0 and w>0 and h>0:
						# Transforming the face
						faceAligned = fa.align(image, image, rect)
						# Calculating blur value.
						blur = cv2.Laplacian(faceAligned, cv2.CV_64F).var()
						# Comparing the blur value to the threshold.
						if blur > blur_limit:
							x_images.append(faceAligned[None])
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