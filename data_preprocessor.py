##########################################################################
# Creators: Lajos Bodo, Szilard Kosa
# Description: The program opens the dataset and preprocesses it.
##########################################################################

# This library let's us work with hdf5 format files.
import tables
# Fundamental package for scientific computing with Python.
import numpy as np

# Opening the dataset.
hdf5_path = "dataset.hdf5"
hdf5_file = tables.open_file(hdf5_path, mode='r')
print("Opening %s ."%(hdf5_path))

# Defining the number of pictures loaded for each person.
picture_per_person = 10

# Lists for storing the data from the dataset.
X = []
Y = []

# Defining variables needed for the algorithm.
data_num = hdf5_file.root.images.shape[0]
prev_label = hdf5_file.root.labels[0]
count = 0

# Reads the defined amount of pictures for each person.
for i in range(data_num):
    label = hdf5_file.root.labels[i]
    print("Loading: %.3f %%"%(float(i)/float(data_num-1)*100),end="\r", flush=True)
    if label != prev_label:
        count = 0

    if count < picture_per_person:
        X.append(hdf5_file.root.images[i])
        Y.append(label)
        count += 1
     
    prev_label = label

# Closing the file.
hdf5_file.close()
print()
print("HDF5 file closed.")

# Defining the variables to split the dataset.
valid_split = 0.1
test_split = 0.1
# Calculating the splitting indexes.
nb_samples = len(Y)
valid_index = int(nb_samples*(1-valid_split-test_split)//picture_per_person)*picture_per_person
test_index = int(nb_samples*(1-test_split)//picture_per_person)*picture_per_person

# Splitting and standardizing the dataset.
X_train = [i/255 for i in X[0:valid_index]]
Y_train = [i/255 for i in Y[0:valid_index]]
X_valid = [i/255 for i in X[valid_index:test_index]]
Y_valid = [i/255 for i in Y[valid_index:test_index]]
X_test  = [i/255 for i in X[test_index:]]
Y_test  = [i/255 for i in Y[test_index:]]
print("Number of elements in X_train: %i" %(len(X_train)))
print("Number of elements in Y_train: %i" %(len(Y_train)))
print("Number of elements in X_valid: %i" %(len(X_valid)))
print("Number of elements in Y_valid: %i" %(len(Y_valid)))
print("Number of elements in X_test: %i" %(len(X_test)))
print("Number of elements in Y_test: %i" %(len(Y_test)))

################################
# Preproccessing training data #
################################
# For the training here we group the data into batches.
# A batch contains a defined amount of pictures for the specified number of people.
X_train_batch = []
Y_train_batch = []
# Defining the number of persons included in a batch.
person_per_batch = 10
# Calculating the number of pictures in a batch.
picture_per_batch = picture_per_person*person_per_batch
# Calculating number of batches.
batch_number = int(len(Y_train)//picture_per_batch)
print("=================== Training data ===================")
print("The number of batches used in training: {}".format(batch_number))
for i in range(batch_number):
    X_train_batch.append(X_train[i*picture_per_batch:(i+1)*picture_per_batch])
    Y_train_batch.append(Y_train[i*picture_per_batch:(i+1)*picture_per_batch])

##################################
# Preproccessing validation data #
##################################
# Creating lists for the validation data.
# The positives will contain pairs from the same person.
# The negatives will contain pairs from two different persons.
X_valid_positives = []
Y_valid_positives = []
X_valid_negatives = []
Y_valid_negatives = []

# The validation data is cut in half annd loaded into the lists defined above.
half_index_valid = int((0.5*len(Y_valid)//picture_per_person)*picture_per_person)
X_valid_positives = X_valid[0:half_index_valid]
Y_valid_positives = Y_valid[0:half_index_valid]
print("=================== Validation data ===================")
print("Number of elements in X_valid_positives: %i" %(len(X_valid_positives)))
print("Number of elements in Y_valid_positives: %i" %(len(Y_valid_positives)))

# The negatives list is made with the following technique:
# In the first half of the list, to every second place we insert a random element from the second half.
# This way every two following pictures are from different persons.
half_index_valid_negatives = half_index_valid + int((0.5*len(Y_valid[half_index_valid:])//picture_per_person)*picture_per_person)
x_valid_negatives = X_valid[half_index_valid:half_index_valid_negatives]
y_valid_negatives = Y_valid[half_index_valid:half_index_valid_negatives]
randperm = np.random.permutation(len(Y_valid[half_index_valid_negatives:]))
for i in range(int(len(x_valid_negatives)/picture_per_person)):
  for j in range(picture_per_person):
    X_valid_negatives.append(x_valid_negatives[i*picture_per_person+j])
    X_valid_negatives.append(X_valid[randperm[i*picture_per_person+j]+half_index_valid_negatives])
    Y_valid_negatives.append(y_valid_negatives[i*picture_per_person+j])
    Y_valid_negatives.append(Y_valid[randperm[i*picture_per_person+j]+half_index_valid_negatives])
print("Number of elements in X_valid_negatives: %i" %(len(X_valid_negatives)))
print("Number of elements in Y_valid_negatives: %i" %(len(Y_valid_negatives)))


############################
# Preproccessing test data #
############################
# Creating lists for the test data.
# The positives will contain pairs from the same person.
# The negatives will contain pairs from two different persons.
X_test_positives = []
Y_test_positives = []
X_test_negatives = []
Y_test_negatives = []

# The test data is cut in half annd loaded into the lists defined above.
half_index_test = int((0.5*len(X_test)//picture_per_person)*picture_per_person)
X_test_positives = X_test[0:half_index_test]
Y_test_positives = Y_test[0:half_index_test]
print("=================== Test data ===================")
print("Number of elements in X_test_positives: %i" %(len(X_test_positives)))
print("Number of elements in Y_test_positives: %i" %(len(Y_test_positives)))

# The negatives list is made with the following technique:
# In the first half of the list, to every second place we insert a random element from the second half.
# This way every two following pictures are from different persons.
half_index_test_negatives = half_index_test + int((0.5*len(Y_test[half_index_test:])//picture_per_person)*picture_per_person)
x_test_negatives = X_test[half_index_test:half_index_test_negatives]
y_test_negatives = Y_test[half_index_test:half_index_test_negatives]
randperm = np.random.permutation(len(Y_test[half_index_test_negatives:]))
for i in range(int(len(x_test_negatives)/picture_per_person)):
  for j in range(picture_per_person):
    X_test_negatives.append(x_test_negatives[i*picture_per_person+j])
    X_test_negatives.append(X_test[randperm[i*picture_per_person+j]+half_index_test_negatives])
    Y_test_negatives.append(y_test_negatives[i*picture_per_person+j])
    Y_test_negatives.append(Y_test[randperm[i*picture_per_person+j]+half_index_test_negatives])
print("Number of elements in X_test_negatives: %i" %(len(X_test_negatives)))
print("Number of elements in Y_test_negatives: %i" %(len(Y_test_negatives)))