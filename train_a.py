from scipy import ndimage as img
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import path as Path
import numpy as np
import os, sys
import cv2
import re
from keras import backend as K
K.set_image_dim_ordering('tf')

#Get the DataSet
DatasetPath = []
DatasetPath.append(os.getcwd() + '/DataSet/hpatel')
DatasetPath.append(os.getcwd() + '/DataSet/nsmith')
DatasetPath.append(os.getcwd() + '/DataSet/agupta')
DatasetPath.append(os.getcwd() + '/DataSet/nvarun')

#variable to store data and labels
imageData = []
imageLabels = []



def splitall(path):
    """Split a path into all of its parts.
    
    From: Python Cookbook, Credit: Trent Mick
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

#load images from this particular folder
def load_images(folder):
    images = []
    currentLabel = (folder)	
    currentLabel = splitall(currentLabel)
    currentLabel = (currentLabel[-1])#.stripext()
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))
#        print img.shape
        if img is not None:
            imageData.append(img)  
            imageLabels.append(currentLabel)  

    print 'number of images in dataset : ', len(imageData)
    print 'number of images in dataset : ', len(imageLabels)
    return images

for currentPath in DatasetPath:
    load_images(currentPath)

    
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(imageLabels)
encoded_Y = encoder.transform(imageLabels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print '-------------------------------------------------'
print '-------------------------------------------------'
print 'number of images in dataset : ', len(imageData)
print 'number of labels in dataset : ', len(dummy_y)

#for j in range(0, len(dummy_y)):
#	print dummy_y[j], ' : ' , imageLabels[j]

    #if picture.endswith(".jpeg"):	
#        imgRead = cv2.imread(i, 0)
#        print(imgRead)
#        imageData.append(imgRead)
#        labelRead = int(os.path.split(i)[1].split("_")[0]) - 1
#        imageLabels.append(labelRead)


# split randomly the photos into 2 parts, 
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(dummy_y), train_size=0.8, random_state = 4)

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(y_train) 
Y_test = np.array(y_test)

#print np.unique(np.array(dummy_y))
# how many people for this model

nb_classes = dummy_y.shape[1]
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test  = np_utils.to_categorical(Y_test, nb_classes)

# for tensorflow backend, it's (nb_of_photo, size, size, channel)
# for theanos backend, it's (channel, nb_of_photo, size, size)
X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])

# first layer of model.
input_shape = (X_train.shape[0],X_train.shape[1], X_train.shape[2])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print (X_train.shape)
print (Y_train.shape)
X_test  = X_test.transpose(1,2,3,0)
X_train  = X_train.transpose(1,2,3,0)
print (X_train.shape)
print (Y_train.shape)
#for i in Y_train:
#	print i

model = Sequential()

#layer 1 : convolutional
#layer 2 : convolutional
#layer 3 : pooling
#layer 4 : dropout
#layer 5 : convolutional
#layer 6 : convolutional
#layer 7 : maxpool
#layer 8 : dropout
#layer 9 : fully-connected

#print input_shape

model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=20,
                 verbose=1, validation_data=(X_test, Y_test))
exit()
# save the trained model.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# and use the 20% data as we have already splited to test the new model
scores = model.evaluate(X_test, Y_test, verbose=0)
print (scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



