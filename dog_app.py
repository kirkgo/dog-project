
# coding: utf-8

# Import Dog Dataset
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# Import Human Dataset
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

# Detect Humans
import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# Human Face Detector
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Assess the Human Face Detector
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
humans_list = []
dogs_list = []

for file in human_files_short:
    humans_list.append(int(face_detector(file)))

humans_score = np.mean(humans_list)

for file in dog_files_short:
    dogs_list.append(int(face_detector(file)))
    
dogs_score = np.mean(dogs_list)
        
print("% humans detected:", humans_score)
print("% dogs detected:", dogs_score)

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# Pre-process the Data
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Making Predictions with ResNet-50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# Dog Detector
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# Assess the Dog Detector
humans_list = []
dogs_list = []

for file in human_files_short: 
    humans_list.append(int(dog_detector(file)))
    
humans_score = np.mean(humans_list)

for file in dog_files_short: 
    dogs_list.append(int(dog_detector(file)))
    
dogs_score = np.mean(dogs_list)

print("% humans with dog detection:", humans_score)
print("% dogs with dog dectection:", dogs_score)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# Model Architecture
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()

# First layer
model.add(Conv2D(16, 5, 5, input_shape=(224,224,3), subsample=(2,2), border_mode='same', name='conv1'))
model.add(Activation('relu'))

# First pool
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), name='pool1'))

# Second layer
model.add(Conv2D(32, 5, 5,border_mode='same', subsample=(2,2), name='conv2'))
model.add(Activation('relu', name='relu2'))

# Second pool
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

# Third layer
model.add(Conv2D(64, 3, 3, border_mode='same', name='conv3'))
model.add(Activation('relu'))

# Third pool
model.add(MaxPooling2D(pool_size= (2,2), name='pool3'))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(133, activation='softmax', name='output'))
model.summary()


# Compile the Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the Model
from keras.callbacks import ModelCheckpoint  

epochs = 8
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# Load the Model with the Best Validation Loss
model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# Test the Model
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# Obtain Bottleneck Features
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

# Model Architecture
# The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output 
# of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where 
# the latter contains one node for each dog category and is equipped with a softmax.
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()

# Compile the Model
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the Model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

# Load the Model with the Best Validation Loss
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

# Test the Model 
# Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  
# We print the test accuracy below.

# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# Predict Dog Breed with the Model
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']


# Model Architecture
from keras.layers.core import Reshape

VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_model.add(Dense(512, activation='relu'))
VGG19_model.add(Dense(133, activation='softmax'))
VGG19_model.summary()

# Compile the Model
adam = Adam(lr=0.0007, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
VGG19_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the Model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', verbose=1, save_best_only=True)
VGG19_model.fit(train_VGG19, train_targets, validation_data=(valid_VGG19, valid_targets), epochs=8, batch_size=64, callbacks=[checkpointer], verbose=1)

# Load the Model with the Best Validation Loss
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

# Test the Model
# Calculate classification accuracy on the test dataset.
VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# Predict Dog Breed with the Model
from extract_bottleneck_features import *
from keras.applications.vgg19 import preprocess_input, decode_predictions

def VGG19_predict_dog_breed(img_path):
    bottleneck_feature = extract_VGG19(preprocess_input(path_to_tensor(img_path)))
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

# Test dog breed
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg

def my_predictor(img_path):
    img= mpimg.imread(img_path)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

    if dog_detector(img_path) == True:
        print("Found a dog! The breed is:", VGG19_predict_dog_breed(img_path))
    elif face_detector(img_path) == True:
         print("Wow! It's a human! He resemble the breed:", VGG19_predict_dog_breed(img_path))
    else:
        dog_breed = None
        print("Ooops! It's not a human or dog...")


# Run algorithm
for infile in sorted(glob("testImages/*")):
     my_predictor(infile)

