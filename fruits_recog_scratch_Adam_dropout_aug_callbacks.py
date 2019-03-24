import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from keras import layers
from keras.models import Sequential
from keras import optimizers
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, regularizers, Flatten, Dense
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import os, os.path
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
import numpy as np 

start_t = timer()


# declare the paths of downloaded folder; Training, Validation and Test folder

# Local machine
# training_dir = '/Users/bathuy/Downloads/fruits-360/Training'
# validation_dir = '/Users/bathuy/Downloads/fruits-360/Validation'
# test_dir = '/Users/bathuy/Downloads/fruits-360/Test'

# GCP compute
training_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Training'
validation_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Validation'
test_dir = '/home/nhaclap_phan/DeepLearning/fruits-360/Test'

# define the parameters
saved_path = os.path.join(os.getcwd(), 'saved_models')
try:
    os.mkdir(saved_path)
except:
    pass
    
model_name = 'CNN_scratch_dropout.h5'
model_name_aug = 'CNN_scratch_Adam_dropout_aug_callbacks.h5'
model_weights = 'CNN_scratch_dropout_aug_weights.h5'
model_weights_data_aug = 'CNN_scratch_Adam_dropout_aug_callbacks_weights.h5'
num_classes = 95
data_augmentation = True

#--------------------------------------------------------------------------

# define callbacks list
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=10),
    keras.callbacks.ModelCheckpoint(filepath='callbacks_model.h5', monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
]
# define the training model
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(95))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt ,metrics=['accuracy'])

model.summary()
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


# train the model
num_epochs = 100
batch_size = 1024

# define data generator for training set and validation set
print("Training set:")
train_generator = train_datagen.flow_from_directory(training_dir, batch_size=batch_size, target_size=(100,100))
print("Validation set:")
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=batch_size, target_size=(100, 100))
print("Test set:")
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=batch_size, target_size=(100,100))

history = model.fit_generator(train_generator,
                            steps_per_epoch=48905//batch_size, epochs=num_epochs,
                            callbacks=callbacks_list,
                            validation_data=validation_generator,
                            validation_steps=8205//batch_size)

if data_augmentation:
    model.save(os.path.join(saved_path, model_name_aug))
    model.save_weights(os.path.join(saved_path, model_weights_data_aug))
else:
    model.save(os.path.join(saved_path, model_name))
    model.save_weights(os.path.join(saved_path, model_weights))

# --------display history--------
# list all data in history
print(history.history.keys())

test_loss, test_acc = model.evaluate_generator(test_generator, steps=8216//batch_size, verbose=1)
print('Test accuracy = {:.4f}'.format(test_acc))
print('Test loss = {:.4f}'.format(test_loss))

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
plt.savefig(os.path.join(saved_path, 'train_test_Adam_accuracy_dropout_aug_callbacks.png'))
plt.clf()  # clear figure

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title(('Training and validation loss'))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(os.path.join(saved_path, 'train_test_Adam_loss_dropout_aug_callbacks.png'))
plt.clf()

stop_t = timer()
print('Elapsed time: {}'.format(stop_t - start_t))

