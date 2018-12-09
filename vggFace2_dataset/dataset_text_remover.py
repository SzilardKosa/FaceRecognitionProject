#################################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program process and merges the datasets created by the
# 'pictures_to_hdf5.py' into one dataset.
# It filters out the pictures with text on them and saves only those persons who
# has at least a predefined number of pictures. 
#################################################################################

# This library let's us work with hdf5 format files.
import tables
# Used for image processing and face detection.
import cv2

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
data_shape = (0, 128, 128)
hdf5_path = 'dataset_without_text.hdf5'
hdf5_write = tables.open_file(hdf5_path, mode='w')
storage = hdf5_write.create_earray(hdf5_write.root, 'images', img_dtype, shape=data_shape)

# load the pre-trained EAST text detector
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Defining lists needed for the algorithm
y_labels = []
y_storage = []
x_images = []
# The predefined number of pictures for each person saved into the dataset.
num_limit = 40
num = 0

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
			# Check text
			color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			if text_detector(color, 0.3, net, 128, 128) < 1:
				x_images.append(image[None])
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