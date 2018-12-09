##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program creates a HDF5 dataset from the lfw database.
# The dataset's structure is defined by the pairs.txt file.
##########################################################################

# This library let's us work with hdf5 format files.
import tables
# Used for image processing and face detection.
import cv2

# This function is used to create the path to the image.
# directory: the name of the directory where the invidula person's directories are located (default: lfw)
# person: the name of the person pictured on the image
# img_number: specifies which image of the given person
# file_type: the file's extension
def create_path(directory, person, img_number, file_type='jpg'):
	# The image numbering is the following: personname_xxxx.jpg
	# The img_number parameter is transformed to the necessary format
	if int(img_number)<10:
		img_number = '000' + img_number
	elif int(img_number)<100:
		img_number = '00' + img_number
	elif int(img_number)<1000:
		img_number = '0' + img_number

	path = directory + '/' + person +'/' + person + '_' + img_number + '.' + file_type
	return path

# This function is used to load and process the image
# line: a line from the pairs.txt, which contains the current pair's data
# directory: the name of the directory where the invidual person's directories are located (default: lfw)
# element: the line is split into parts, the element refers to one part
# same: True, if the images in the pair are belonging to the same person
def process_image(line, directory, element, same):
	# Split the line
	if same:
		person = line.split()[0]
	else:
		person = line.split()[element - 1]
	img_num = line.split()[element]
	# Create path to the image
	path = create_path(directory, person, img_num)
	# Load the image
	image = cv2.imread(path)
	# Turn the image into a greyscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# The img_number parameter is transformed to the necessary format, for the correct labeling
	if int(img_num)<10:
		img_num = '000' + img_num
	elif int(img_num)<100:
		img_num = '00' + img_num
	elif int(img_num)<1000:
		img_num = '0' + img_num
	# Store the image
	# The label contains the specific number of the image for later use in the 'set_converter.py' program
	storage.append(gray[None])
	y_labels.append(person + '_' + img_num)

# Load the overlap.txt
overlaps_file = open('overlap.txt', 'r')
overlaps = []
# process the data from the overlap.txt
for line in overlaps_file:
	overlaps.append(line.split()[0])

# Creating the HDF5 file
img_dtype = tables.UInt8Atom()
data_shape = (0, 250, 250)
hdf5_path = 'dataset_lfw_set.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='w')
storage = hdf5_file.create_earray(hdf5_file.root, 'images', img_dtype, shape=data_shape)
y_labels = []

# The name of the directory, where the data is stored
directory = 'lfw'

# Open the pairs.txt
pairs_file = open('pairs.txt', 'r')

# Create lists for storing the pairs
matched_pairs = []
mismatched_pairs = []

# Variables for the iteration
line_count = 0
current_set = 0
number_of_overlaps = 0
good_images = 0
change_index = []
# Iterate through the pairs file and make the pairs
for line in pairs_file:
	if line_count % 600 == 0:
		current_set += 1
	# Get number of sets and the pictures per set from the first line
	if line_count == 0:
		number_of_sets = int(line.split()[0])
		pairs_per_sets = int(line.split()[1])
	else:
		# Determine if the current line contains a matched or mismatched pair
		# Matched pair
		if 0 < (line_count - (2 * pairs_per_sets)*(current_set-1)) % (2 * pairs_per_sets) < (pairs_per_sets + 1):
			person = line.split()[0]
			# Check if the person is one of the overlaps with the train data
			if person not in overlaps:
				process_image(line, directory, 1, True)
				process_image(line, directory, 2, True)
				good_images += 1
			else:
				number_of_overlaps += 1
		# Mismatched pair
		else:
			person1 = line.split()[0]
			person2 = line.split()[2]
			# Check if either person is one of the overlaps with the train data
			if (person1 not in overlaps) and (person2 not in overlaps):
				process_image(line, directory, 1, False)
				process_image(line, directory, 3, False)
				good_images += 1
			else:
				number_of_overlaps += 1
	# Each set is made of 300 matched and 300 mismatched pair
	# Due to the removal of the overlaps, the sets contains less number of pairs
	# Save the indexes where there is a change between a matched and mismatched pair
	if line_count % 300 == 0 and line_count!=0:
		change_index.append(good_images)
	line_count += 1

# Saving the labels in the HDF5 file.
hdf5_file.create_array(hdf5_file.root, 'labels', y_labels)
# Closing the file.
hdf5_file.close()
print("HDF5 file closed.")
print('number of overlaps: {}'.format(number_of_overlaps))
print('number of good images: {}'.format(good_images))

# Create and save the index data into a txt file for later use
with open('set_index.txt', 'w') as f:
    for number in change_index:
        f.write("%s\n" % number)
f.close()