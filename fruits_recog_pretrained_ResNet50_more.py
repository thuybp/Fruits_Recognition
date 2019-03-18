import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from keras import layers
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, regularizers, Flatten, Dense
from keras.applications import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import os, os.path
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
import numpy as np 

K.set_learning_phase(1)

CNN_ResNet = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(100, 100, 3))

start_t = timer()


# declare the paths of downloaded folder; Training, Validation and Test folder

# Local link
# training_dir = '/Users/bathuy/Downloads/fruits-360/Training'
# validation_dir = '/Users/bathuy/Downloads/fruits-360/Validation'
# test_dir = '/Users/bathuy/Downloads/fruits-360/Test'

# GCP link
training_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Training'
validation_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Validation'
test_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Test'

# AWS link
# training_dir = '/home/ubuntu/DeepLearning/Fruit-Images-Dataset/Training'
# validation_dir = '/home/ubuntu/DeepLearning/Fruit-Images-Dataset/Validation'
# test_dir = '/home/ubuntu/DeepLearning/Fruit-Images-Dataset/Test'

# define the parameters
saved_path = os.path.join(os.getcwd(), 'saved_models')
try:
    os.mkdir(saved_path)
except:
    pass

# define filenames for saving the model
model_name_aug = 'CNN_pretrained_ResNet50_aug.h5'
model_weights_data_aug = 'CNN_pretrained_ResNet50_aug_weights.h5'
model_name = 'CNN_pretrained_ResNet50.h5'
model_weights = 'CNN_pretrained_ResNet50_weights.h5'
fine_model_name_aug = 'CNN_fine_pretrained_ResNet50_aug.h5'
fine_model_weights_data_aug = 'CNN_fine_pretrained_ResNet50_aug_weights.h5'
fine_model_name = 'CNN_fine_pretrained_ResNet50.h5'
fine_model_weights = 'CNN_fine_pretrained_ResNet50_weights.h5'
num_classes = 95
data_augmentation = True

# define callbacks list
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=10),
    keras.callbacks.ModelCheckpoint(
        filepath='callbacks_model.h5', monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5),
]
#--------------------------------------------------------------------------
# # define the classifier
# model = Sequential()
# model.add(CNN_ResNet)
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Dense(95))
# model.add(Activation('softmax'))

model = load_model(os.path.join(saved_path, model_name_aug))
model.load_weights(os.path.join(saved_path, model_weights_data_aug))

opt = keras.optimizers.RMSprop(lr=0.0001)

#---------------------------------------------------------------------------------
# define data generator
if data_augmentation:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
else:
    train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


batch_size = 1024
# train the model
num_epochs = 100

# define data generator for training set and validation set
print("Training set:")
train_generator = train_datagen.flow_from_directory(
    training_dir, batch_size=batch_size, target_size=(100, 100))
print("Validation set:")
validation_generator = test_datagen.flow_from_directory(
    validation_dir, batch_size=batch_size, target_size=(100, 100))
print("Test set:")
test_generator = test_datagen.flow_from_directory(
    test_dir, batch_size=batch_size, target_size=(100, 100))

CNN_ResNet.trainable = False
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=48905//batch_size, epochs=num_epochs,
                              callbacks=callbacks_list,
                              validation_data=validation_generator,
                              validation_steps=8205//batch_size)

# save the model
if data_augmentation:
    model.save(os.path.join(saved_path, model_name_aug))
    model.save_weights(os.path.join(saved_path, model_weights_data_aug))
else:
    model.save(os.path.join(saved_path, model_name))
    model.save_weights(os.path.join(saved_path, model_weights))

# load the model
# fine_model = load_model(os.path.join(saved_path, model_name_aug))
# fine_model.load_weights(os.path.join(saved_path, model_weights_data_aug))

# # unfreeze the top layers of ResNet50 and train the model again
# CNN_ResNet.trainable = True
# set_trainable = False
# train_layers = ['res5a_branch2a', 'bn5a_branch2a',
#                 'res5a_branch2b', 'bn5a_branch2b', 'res5a_branch2c', 'res5a_branch1',
#                 'bn5a_branch2c', 'bn5a_branch1', 'res5b_branch2a', 'bn5b_branch2a',
#                 'res5b_branch2b', 'bn5b_branch2b', 'res5b_branch2c', 'bn5b_branch2c',
#                 'res5c_branch2a', 'bn5c_branch2a', 'res5c_branch2b', 'bn5c_branch2b',
#                 'res5c_branch2c', 'bn5c_branch2c', ]

# for layer in CNN_ResNet.layers:
#     if layer.name in train_layers:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# fine_model.summary()
# fine_model.compile(loss='categorical_crossentropy',
#               optimizer=opt, metrics=['accuracy'])

# history = fine_model.fit_generator(train_generator,
#                               steps_per_epoch=48905//batch_size, epochs=num_epochs,
#                               callbacks=callbacks_list,
#                               validation_data=validation_generator,
#                               validation_steps=8205//batch_size)
#--------------------------------------------------------------------------------------
# save the model
# if data_augmentation:
#     fine_model.save(os.path.join(saved_path, fine_model_name_aug))
#     fine_model.save_weights(os.path.join(
#         saved_path, fine_model_weights_data_aug))
# else:
#     fine_model.save(os.path.join(saved_path, fine_model_name))
#     fine_model.save_weights(os.path.join(saved_path, fine_model_weights))

# --------display history--------
# list all data in history
print(history.history.keys())

test_loss, test_acc = model.evaluate_generator(
    test_generator, steps=8216//batch_size, verbose=1)
print('Test accuracy = {:.6f}'.format(test_acc))
print('Test loss = {:.6f}'.format(test_loss))

# summarize history for accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(os.path.join(saved_path, 'train_test_accuracy_ResNet50.png'))
plt.clf()  # clear figure

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title(('Training and validation loss'))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper rigth')
plt.savefig(os.path.join(saved_path, 'train_test_loss_ResNet50.png'))
plt.clf()

stop_t = timer()
print('Elapsed time: {}'.format(stop_t - start_t))
