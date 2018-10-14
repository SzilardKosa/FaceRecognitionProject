##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program goes through the previously created datasets
# and counts the number of images for each person.
# The number is determined from the number of labels.
##########################################################################

# This library let's us work with hdf5 format files.
import tables
# Fundamental package for scientific computing with Python.
import numpy as np

# Opening the first dataset.
hdf5_path = 'dataset_01.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
labels1 = np.array(hdf5_file.root.labels)
print("The labels of %s :"%(hdf5_path))
print(labels1)
hdf5_file.close()
print("HDF5 file closed.")

# Opening the second dataset.
hdf5_path = 'dataset_02.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
labels2 = np.array(hdf5_file.root.labels)
print("The labels of %s :"%(hdf5_path))
print(labels2)
hdf5_file.close()
print("HDF5 file closed.")

# Opening the third dataset.
hdf5_path = 'dataset_03.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
labels3 = np.array(hdf5_file.root.labels)
print("The labels of %s :"%(hdf5_path))
print(labels3)
hdf5_file.close()
print("HDF5 file closed.")

# This array is for storing the number of labels in the corresponding indexes of the array.
pic_numbers = np.zeros(10000)
for label in labels1:
	pic_numbers[label] += 1
for label in labels2:
	pic_numbers[label] += 1
for label in labels3:
	pic_numbers[label] += 1

# Saves the data in a txt file, using an Excel compatible format.
file = open("testfile.txt","w")
for i in range(10000):
	file.write("%d\t%d\n"%(i,pic_numbers[i]))
file.close()
print("Txt file closed.")